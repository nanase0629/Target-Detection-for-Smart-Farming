// main.go
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/http/pprof"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sony/gobreaker"
	"github.com/tejzpr/ordered-concurrently/v3"
	"gocv.io/x/gocv"
)

const (
	// 指向 Python 服务的 URL
	pythonServiceURL = "http://localhost:8001/process_images_batch/"
	// 允许客户端访问的视频目录
	ALLOWED_VIDEO_DIR = "./videos"

	// --- 性能调优参数 ---
	// 一次批量发送给 Python 服务的帧数。这个值可以根据你的GPU性能调整，通常与Python端的批处理能力匹配。
	BATCH_SIZE = 8
	// 如果帧不够一个批次，等待此时间后强制发送。
	BATCH_TIMEOUT = 200 * time.Millisecond
	// 并发调用 Python 服务的协程池大小。通常设置为 GPU 数量的 1-2 倍。
	WORKER_POOL_SIZE = 4
	// 存放待处理原始帧的通道缓冲大小
	RAW_FRAMES_CHAN_SIZE = 100

	// 设定一个目标播放帧率
	targetFPS = 20.0

	// --- WebSocket 设置 ---
	writeWait      = 10 * time.Second    // 允许写入消息的时间
	pongWait       = 60 * time.Second    // 等待 pong 消息的时间
	pingPeriod     = (pongWait * 9) / 10 // 发送 ping 的周期
	maxMessageSize = 512                 // 允许从 peer 接收的最大消息大小
)

// Server 封装了应用的所有状态和依赖
type Server struct {
	hub *Hub

	// 视频处理管道的状态
	processingLock     sync.Mutex
	isProcessing       bool
	pipelineCancelFunc context.CancelFunc
	pipelineWg         *sync.WaitGroup
	currentVideoPath   string

	// 外部服务依赖
	httpClient     *http.Client
	circuitBreaker *gobreaker.CircuitBreaker

	// 监控
	processedFrames atomic.Int64
}

// NewServer 创建并初始化一个新的 Server 实例
func NewServer() *Server {
	httpClient := &http.Client{Timeout: 30 * time.Second} // 增加了超时时间以适应可能的长时间批处理
	st := gobreaker.Settings{
		Name:        "python-service-cb",
		MaxRequests: 3,
		Interval:    30 * time.Second,
		Timeout:     30 * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures > 5
		},
		OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
			log.Printf("断路器 '%s' 状态变更: 从 '%s' 到 '%s'", name, from, to)
		},
	}

	return &Server{
		hub:            newHub(),
		httpClient:     httpClient,
		circuitBreaker: gobreaker.NewCircuitBreaker(st),
		pipelineWg:     &sync.WaitGroup{},
	}
}

// Run 启动服务器的所有后台服务 (Hub, 吞吐量监控等)
func (s *Server) Run(ctx context.Context) {
	go s.hub.Run(ctx)
	go s.monitorThroughput(ctx)
	log.Println("核心服务（Hub, 监控）已启动。")
}

// WebSocket 升级器
var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // 开发模式下允许所有来源的连接
	},
}

// Client 是 WebSocket 客户端和 Hub 之间的中间人
type Client struct {
	hub  *Hub
	conn *websocket.Conn
	send chan []byte // 客户端的带缓冲出站消息通道
}

// Hub 管理 WebSocket 连接和广播的中央管理器
type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.Mutex
}

func newHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

// Run 启动 Hub 的事件循环
func (h *Hub) Run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			h.mu.Lock()
			for client := range h.clients {
				close(client.send)
				delete(h.clients, client)
			}
			h.mu.Unlock()
			log.Println("Hub: 上下文已取消，正在关闭。")
			return
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			log.Printf("WebSocket客户端已连接: %s. 当前总连接数: %d", client.conn.RemoteAddr(), len(h.clients))
			h.mu.Unlock()
		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
				log.Printf("WebSocket客户端已断开: %s. 当前总连接数: %d", client.conn.RemoteAddr(), len(h.clients))
			}
			h.mu.Unlock()
		case message := <-h.broadcast:
			h.mu.Lock()
			// 使用非阻塞发送，避免单个慢客户端阻塞整个广播
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					close(client.send)
					delete(h.clients, client)
					log.Printf("客户端 %s 发送通道已满，已断开连接。", client.conn.RemoteAddr())
				}
			}
			h.mu.Unlock()
		}
	}
}

// readPump 将消息从 websocket 连接泵送到 hub
func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()
	c.conn.SetReadLimit(maxMessageSize)
	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})
	for {
		if _, _, err := c.conn.ReadMessage(); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket 读取错误: %v", err)
			}
			break
		}
	}
}

// writePump 将消息从 hub 泵送到 websocket 连接
func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()
	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.BinaryMessage, message); err != nil {
				return
			}
		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// handleWebSocket 为给定的 Server 实例处理 WebSocket 请求
func (s *Server) handleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Println("WebSocket 升级失败:", err)
		return
	}
	client := &Client{hub: s.hub, conn: conn, send: make(chan []byte, 256)}
	client.hub.register <- client

	go client.writePump()
	go client.readPump()
}

// --- Python 服务相关的结构体 ---

type PythonBatchResponse struct {
	Results []ProcessResult `json:"results"`
}

type ProcessResult struct {
	FinalImageBase64 string      `json:"final_image_base64"`
	DetectedHumans   [][]float64 `json:"detected_humans"`   // 可选，用于调试
	DetectedChickens [][]float64 `json:"detected_chickens"` // 可选，用于调试
}

type PythonProcessingParams struct {
	BlurMethod  string  `form:"blur_method"`
	HumanConf   float64 `form:"human_conf"`
	ChickenConf float64 `form:"chicken_conf"`
}

// --- main 函数 ---
func main() {
	runtime.SetBlockProfileRate(1)

	if err := os.MkdirAll(ALLOWED_VIDEO_DIR, os.ModePerm); err != nil {
		log.Fatalf("无法创建视频目录 '%s': %v", ALLOWED_VIDEO_DIR, err)
	}
	log.Printf("视频文件将从 '%s' 目录读取。", ALLOWED_VIDEO_DIR)

	rootCtx, rootCancel := context.WithCancel(context.Background())
	defer rootCancel()

	server := NewServer()
	server.Run(rootCtx)

	router := gin.Default()

	// 提供前端页面
	router.StaticFile("/", "../frontend/index.html")

	// API 路由
	api := router.Group("/api")
	{
		api.POST("/start_monitoring", server.handleStartMonitoring)
		api.POST("/stop_monitoring", server.handleEndMonitoring) // Renamed from end_monitoring for consistency
		api.GET("/status", server.handleStatus)
	}

	// WebSocket 路由
	router.GET("/ws", server.handleWebSocket) // 简化路径

	// Pprof 路由
	pprofGroup := router.Group("/debug/pprof")
	{
		pprofGroup.GET("/", gin.WrapH(http.HandlerFunc(pprof.Index)))
		pprofGroup.GET("/cmdline", gin.WrapH(http.HandlerFunc(pprof.Cmdline)))
		pprofGroup.GET("/profile", gin.WrapH(http.HandlerFunc(pprof.Profile)))
		pprofGroup.GET("/symbol", gin.WrapH(http.HandlerFunc(pprof.Symbol)))
		pprofGroup.GET("/trace", gin.WrapH(http.HandlerFunc(pprof.Trace)))

		pprofGroup.GET("/heap", gin.WrapH(pprof.Handler("heap")))
		pprofGroup.GET("/goroutine", gin.WrapH(pprof.Handler("goroutine")))
		pprofGroup.GET("/threadcreate", gin.WrapH(pprof.Handler("threadcreate")))
		pprofGroup.GET("/block", gin.WrapH(pprof.Handler("block")))
	}

	log.Println("Go 服务已启动，监听端口 :8080. 请访问 http://localhost:8080")
	if err := router.Run(":8080"); err != nil {
		log.Fatalf("无法启动 Gin 服务器: %v", err)
	}
}

// --- API 处理 ---

type StartMonitoringRequest struct {
	VideoPath   string  `json:"video_path" binding:"required"`
	BlurMethod  string  `json:"blur_method"`
	HumanConf   float64 `json:"human_conf"`
	ChickenConf float64 `json:"chicken_conf"`
}

type StatusResponse struct {
	IsProcessing bool   `json:"is_processing"`
	VideoPath    string `json:"video_path,omitempty"`
}

func (s *Server) handleStatus(c *gin.Context) {
	s.processingLock.Lock()
	defer s.processingLock.Unlock()
	c.JSON(http.StatusOK, StatusResponse{
		IsProcessing: s.isProcessing,
		VideoPath:    s.currentVideoPath,
	})
}

func (s *Server) handleStartMonitoring(c *gin.Context) {
	s.processingLock.Lock()
	if s.isProcessing {
		s.processingLock.Unlock()
		c.JSON(http.StatusConflict, gin.H{"error": "另一个视频正在处理中，请先停止。"})
		return
	}

	var req StartMonitoringRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.processingLock.Unlock()
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求参数: " + err.Error()})
		return
	}

	safeVideoName := filepath.Base(req.VideoPath)
	safeVideoPath, err := secureFilePath(ALLOWED_VIDEO_DIR, safeVideoName)
	if err != nil {
		s.processingLock.Unlock()
		c.JSON(http.StatusBadRequest, gin.H{"error": "视频文件路径无效或不被允许: " + err.Error()})
		return
	}
	if _, err := os.Stat(safeVideoPath); os.IsNotExist(err) {
		s.processingLock.Unlock()
		c.JSON(http.StatusNotFound, gin.H{"error": "视频文件未找到: " + safeVideoPath})
		return
	}

	s.isProcessing = true
	s.currentVideoPath = safeVideoName
	var pipelineCtx context.Context
	pipelineCtx, s.pipelineCancelFunc = context.WithCancel(context.Background())
	s.pipelineWg = &sync.WaitGroup{}
	s.processingLock.Unlock()

	log.Printf("开始监控: %s (模糊: %s, 人体阈值: %.2f, 二次阈值: %.2f)",
		safeVideoPath, req.BlurMethod, req.HumanConf, req.ChickenConf)

	rawFramesInChan := make(chan FrameData, RAW_FRAMES_CHAN_SIZE)
	batchWorkInChan := make(chan orderedconcurrently.WorkFunction, RAW_FRAMES_CHAN_SIZE)

	// 创建参数时不再包含模型路径
	params := PythonProcessingParams{
		BlurMethod:  req.BlurMethod,
		HumanConf:   req.HumanConf,
		ChickenConf: req.ChickenConf,
	}

	s.pipelineWg.Add(1)
	go s.videoReader(pipelineCtx, safeVideoPath, rawFramesInChan)

	s.pipelineWg.Add(1)
	go s.batchFramesProcessor(pipelineCtx, rawFramesInChan, batchWorkInChan, params)

	orderedResultChan := orderedconcurrently.Process(
		pipelineCtx,
		batchWorkInChan,
		&orderedconcurrently.Options{PoolSize: WORKER_POOL_SIZE, OutChannelBuffer: 100},
	)

	s.pipelineWg.Add(1)
	go s.dispatcher(pipelineCtx, orderedResultChan)

	c.JSON(http.StatusOK, gin.H{"message": "视频监控已启动。现在可以通过 WebSocket 观看实时流。"})
}

func (s *Server) handleEndMonitoring(c *gin.Context) {
	s.processingLock.Lock()
	defer s.processingLock.Unlock()

	if !s.isProcessing {
		c.JSON(http.StatusOK, gin.H{"message": "当前没有视频正在处理。"})
		return
	}

	log.Println("收到结束监控请求...")
	if s.pipelineCancelFunc != nil {
		s.pipelineCancelFunc()
	}

	done := make(chan struct{})
	go func() {
		s.pipelineWg.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Println("所有处理管道协程已成功关闭。")
	case <-time.After(10 * time.Second):
		log.Println("警告: 等待处理管道协程关闭超时！")
	}

	s.isProcessing = false
	s.currentVideoPath = ""
	s.pipelineCancelFunc = nil

	c.JSON(http.StatusOK, gin.H{"message": "视频监控已停止。"})
}

// --- 视频处理管道 ---

type FrameData struct {
	ID  int64
	Mat gocv.Mat
}

func (s *Server) videoReader(ctx context.Context, videoPath string, rawFramesInChan chan<- FrameData) {
	defer s.pipelineWg.Done()
	defer close(rawFramesInChan)
	defer log.Println("VideoReader: 协程已退出。")

	cap, err := gocv.OpenVideoCapture(videoPath)
	if err != nil {
		log.Printf("VideoReader: 打开视频文件 '%s' 失败: %v", videoPath, err)
		return
	}
	defer cap.Close()

	frameMat := gocv.NewMat()
	defer frameMat.Close()

	var frameID int64 = 0
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if ok := cap.Read(&frameMat); !ok || frameMat.Empty() {
			//log.Println("VideoReader: 无法读取帧或视频已结束。")
			//return
			log.Println("VideoReader: 视频已播放完毕，将从头开始重新播放。")
			cap.Set(gocv.VideoCapturePosFrames, 0)
			// 使用 'continue' 跳过当前循环的剩余部分，
			// 在下一次循环中将尝试读取重置后的第一帧。
			continue
		}

		frameID++
		clonedMat := frameMat.Clone()
		workItem := FrameData{ID: frameID, Mat: clonedMat}

		select {
		case rawFramesInChan <- workItem:
		case <-ctx.Done():
			clonedMat.Close()
			return
		}
	}
}

func (s *Server) batchFramesProcessor(ctx context.Context, rawFramesInChan <-chan FrameData, batchWorkInChan chan<- orderedconcurrently.WorkFunction, params PythonProcessingParams) {
	defer s.pipelineWg.Done()
	defer close(batchWorkInChan)
	defer log.Println("BatchProcessor: 协程已退出。")

	batch := make([]FrameData, 0, BATCH_SIZE)
	ticker := time.NewTicker(BATCH_TIMEOUT)
	defer ticker.Stop()

	sendBatch := func() {
		if len(batch) == 0 {
			return
		}

		// log.Printf("BatchProcessor: 发送一个包含 %d 帧的批次进行处理。", len(batch))
		work := BatchWorkItem{
			ID:     batch[0].ID,
			Frames: batch,
			Params: params,
			server: s,
		}

		select {
		case batchWorkInChan <- work:
		case <-ctx.Done():
			log.Println("BatchProcessor: 发送批次时收到取消信号。")
			for _, frame := range batch {
				frame.Mat.Close()
			}
		}
		batch = make([]FrameData, 0, BATCH_SIZE)
	}

	for {
		select {
		case frame, ok := <-rawFramesInChan:
			if !ok {
				sendBatch()
				return
			}
			batch = append(batch, frame)
			if len(batch) >= BATCH_SIZE {
				sendBatch()
			}
		case <-ticker.C:
			sendBatch()
		case <-ctx.Done():
			for _, frame := range batch {
				frame.Mat.Close()
			}
			return
		}
	}
}

type BatchWorkItem struct {
	ID     int64
	Frames []FrameData
	Params PythonProcessingParams
	server *Server
}

type BatchProcessResult struct {
	OriginalID    int64
	ProcessedData [][]byte
	Err           error
}

// buildMultipartRequest 构建一个 multipart/form-data 请求，将所有图像帧打包到同一个表单字段中
func buildMultipartRequest(ctx context.Context, frames []FrameData, params PythonProcessingParams) (*http.Request, error) {
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)

	// 1. 将所有图片帧写入同一个 form-data 字段 'image_files'
	for i, frame := range frames {
		params := []int{gocv.IMWriteJpegQuality, 80}
		jpegBytes, err := gocv.IMEncodeWithParams(".jpg", frame.Mat, params) // 使用带参数的版本
		if err != nil {
			return nil, fmt.Errorf("frame %d: JPEG 编码失败: %w", frame.ID, err)
		}
		defer jpegBytes.Close()

		part, err := writer.CreateFormFile("image_files", fmt.Sprintf("frame_%d.jpg", i))
		if err != nil {
			return nil, fmt.Errorf("frame %d: 创建 multipart form-file 失败: %w", frame.ID, err)
		}
		if _, err := part.Write(jpegBytes.GetBytes()); err != nil {
			return nil, fmt.Errorf("frame %d: 写入 multipart form-file 失败: %w", frame.ID, err)
		}
	}

	// 2. 添加其他表单字段
	_ = writer.WriteField("blur_method", params.BlurMethod)
	_ = writer.WriteField("human_conf", fmt.Sprintf("%.2f", params.HumanConf))
	_ = writer.WriteField("chicken_conf", fmt.Sprintf("%.2f", params.ChickenConf))

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("关闭 multipart writer 失败: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", pythonServiceURL, body)
	if err != nil {
		return nil, fmt.Errorf("创建 HTTP 请求失败: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req, nil
}

func (w BatchWorkItem) Run(ctx context.Context) interface{} {
	defer func() {
		for _, frame := range w.Frames {
			frame.Mat.Close()
		}
	}()

	var processedImagesBytes [][]byte
	_, cbErr := w.server.circuitBreaker.Execute(func() (interface{}, error) {
		req, err := buildMultipartRequest(ctx, w.Frames, w.Params)
		if err != nil {
			return nil, fmt.Errorf("batch %d: 构建请求失败: %w", w.ID, err)
		}

		resp, httpErr := w.server.httpClient.Do(req)
		if httpErr != nil {
			return nil, fmt.Errorf("batch %d: 请求 Python 服务失败: %w", w.ID, httpErr)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			respBodyBytes, _ := io.ReadAll(resp.Body)
			return nil, fmt.Errorf("batch %d: Python 服务返回错误 %d: %s", w.ID, resp.StatusCode, string(respBodyBytes))
		}

		var apiResp PythonBatchResponse
		if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
			return nil, fmt.Errorf("batch %d: 解码 Python 服务响应失败: %w", w.ID, err)
		}

		if len(w.Frames) != len(apiResp.Results) {
			log.Printf("警告: 发送了 %d 帧，但只收到 %d 个结果。", len(w.Frames), len(apiResp.Results))
		}

		// 【修改】从批处理响应中解码所有图像
		for _, result := range apiResp.Results {
			decodedBytes, b64Err := base64.StdEncoding.DecodeString(result.FinalImageBase64)
			if b64Err != nil {
				log.Printf("batch %d: Base64 解码失败，跳过此帧: %v", w.ID, b64Err)
				continue
			}
			processedImagesBytes = append(processedImagesBytes, decodedBytes)
		}
		return nil, nil
	})

	if cbErr != nil {
		return BatchProcessResult{OriginalID: w.ID, Err: fmt.Errorf("batch %d: Python 服务调用失败(断路器): %w", w.ID, cbErr)}
	}

	return BatchProcessResult{OriginalID: w.ID, ProcessedData: processedImagesBytes, Err: nil}
}

func (s *Server) dispatcher(ctx context.Context, orderedResultChan <-chan orderedconcurrently.OrderedOutput) {
	defer s.pipelineWg.Done()
	defer log.Println("Dispatcher: 协程已退出。")

	// 计算每帧之间的发送间隔发送一个包含
	//frameInterval := time.Duration(1000.0/targetFPS) * time.Millisecond

	log.Printf("Dispatcher: 开始等待处理后的批处理数据... (将以 %.1f FPS 的速率平滑发送)", targetFPS)

	for {
		select {
		case output, ok := <-orderedResultChan:
			if !ok {
				log.Println("Dispatcher: 有序结果通道已关闭。")
				return
			}

			result, ok := output.Value.(BatchProcessResult)
			if !ok {
				log.Printf("Dispatcher: 严重错误, 无法转换结果类型: %T", output.Value)
				continue
			}

			if result.Err != nil {
				log.Printf("Dispatcher: 批次 %d 处理失败: %v", result.OriginalID, result.Err)
				continue
			}

			if len(result.ProcessedData) > 0 {
				// 遍历批处理结果中的每一帧，并以固定间隔广播
				for _, frameData := range result.ProcessedData {
					select {
					case <-ctx.Done(): // 在发送每一帧前都检查是否需要退出
						log.Println("Dispatcher: 在平滑发送期间收到取消信号。")
						return
					default:
						s.hub.broadcast <- frameData
						s.processedFrames.Add(1) // 每发送一帧，计数器加一

						//time.Sleep(frameInterval) // 等待固定的时间间隔
					}
				}
			}

		case <-ctx.Done():
			log.Println("Dispatcher: 收到取消信号，停止分发。")
			return
		}
	}
}

// --- 工具函数 ---
func secureFilePath(allowedBaseDir, requestedPath string) (string, error) {
	cleanedPath := filepath.Clean(requestedPath)
	absAllowedBaseDir, err := filepath.Abs(allowedBaseDir)
	if err != nil {
		return "", fmt.Errorf("无法获取基础目录的绝对路径: %w", err)
	}
	targetPath := filepath.Join(absAllowedBaseDir, cleanedPath)
	absTargetPath, err := filepath.Abs(targetPath)
	if err != nil {
		return "", fmt.Errorf("无法获取目标路径的绝对路径: %w", err)
	}
	if !strings.HasPrefix(absTargetPath, absAllowedBaseDir) {
		return "", fmt.Errorf("请求的路径 '%s' 超出了允许的目录范围", requestedPath)
	}
	return absTargetPath, nil
}

func (s *Server) monitorThroughput(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			count := s.processedFrames.Swap(0)
			fps := float64(count) / 5.0
			log.Printf("[监控] 吞吐量: 过去 5 秒处理了 %d 帧, 当前帧率: %.2f FPS", count, fps)
		case <-ctx.Done():
			log.Println("[监控] 监控吞吐量协程已停止。")
			return
		}
	}
}
