"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

interface MotionData {
  timestamp: number;
  landmarks: Landmark[];
  velocity: number;
  direction: string;
  complexity: number;
}

export default function MotionDirector() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [hasPose, setHasPose] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visualMode, setVisualMode] = useState<"NEON" | "INK" | "CYBER" | "ABSTRACT">("NEON");
  const [showCamera, setShowCamera] = useState(true);
  const [motionHistory, setMotionHistory] = useState<MotionData[]>([]);
  const [stats, setStats] = useState({
    duration: 0,
    complexity: 0,
    avgVelocity: 0,
    peakVelocity: 0,
    directionChanges: 0,
    dominantMotion: "None",
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const motionTrailRef = useRef<{x: number, y: number, age: number}[]>([]);
  const startTimeRef = useRef<number>(0);
  const lastPositionRef = useRef<{x: number, y: number} | null>(null);
  const velocitiesRef = useRef<number[]>([]);

  // Initialize MediaPipe Pose
  useEffect(() => {
    const initPose = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch (err) {
        setError("Failed to load pose detection models");
      }
    };
    initPose();
  }, []);

  // Start camera
  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsCameraActive(true);
        startTimeRef.current = Date.now();
        detectPose();
      }
    } catch (err) {
      setError("Camera access denied. Please allow camera permission.");
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    setIsCameraActive(false);
    setHasPose(false);
    motionTrailRef.current = [];
  };

  // Calculate velocity and direction
  const calculateMotion = (landmarks: Landmark[]) => {
    const nose = landmarks[0];
    const currentPosition = { x: nose.x, y: nose.y };
    
    let velocity = 0;
    let direction = "STILL";
    
    if (lastPositionRef.current) {
      const dx = currentPosition.x - lastPositionRef.current.x;
      const dy = currentPosition.y - lastPositionRef.current.y;
      velocity = Math.sqrt(dx * dx + dy * dy) * 100;
      
      velocitiesRef.current.push(velocity);
      if (velocitiesRef.current.length > 30) velocitiesRef.current.shift();
      
      if (velocity > 0.5) {
        if (dx > 0.01) direction = "RIGHT →";
        else if (dx < -0.01) direction = "LEFT ←";
        else if (dy > 0.01) direction = "DOWN ↓";
        else if (dy < -0.01) direction = "UP ↑";
      }
    }
    
    lastPositionRef.current = currentPosition;
    
    const complexity = Math.min(100, velocity * 50 + Math.random() * 20);
    
    return { velocity, direction, complexity };
  };

  // Detect pose and update visualization
  const detectPose = useCallback(() => {
    if (!videoRef.current || !poseLandmarkerRef.current || !isCameraActive) {
      animationFrameRef.current = requestAnimationFrame(detectPose);
      return;
    }

    const timestamp = Date.now();
    const result = poseLandmarkerRef.current.detectForVideo(videoRef.current, timestamp);

    if (result.poseLandmarks && result.poseLandmarks.length > 0 && result.poseWorldLandmarks) {
      const landmarks = result.poseLandmarks[0];
      const worldLandmarks = result.poseWorldLandmarks[0];
      const { velocity, direction, complexity } = calculateMotion(landmarks);

      setHasPose(true);
      
      const motionData: MotionData = {
        timestamp,
        landmarks,
        velocity,
        direction,
        complexity,
      };
      
      setMotionHistory(prev => {
        const updated = [...prev, motionData].slice(-100);
        
        const duration = (Date.now() - startTimeRef.current) / 1000;
        const avgVelocity = velocitiesRef.current.reduce((a, b) => a + b, 0) / velocitiesRef.current.length || 0;
        const peakVelocity = Math.max(...velocitiesRef.current, 0);
        const directionChanges = updated.filter((m, i) => i > 0 && m.direction !== updated[i-1].direction).length;
        const avgComplexity = updated.reduce((sum, m) => sum + m.complexity, 0) / updated.length;
        
        const directionCounts: Record<string, number> = {};
        updated.forEach(m => {
          directionCounts[m.direction] = (directionCounts[m.direction] || 0) + 1;
        });
        const dominantMotion = Object.entries(directionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "None";
        
        setStats({
          duration,
          complexity: avgComplexity,
          avgVelocity,
          peakVelocity,
          directionChanges,
          dominantMotion,
        });
        
        return updated;
      });

      motionTrailRef.current.forEach(point => {
        point.age += 0.02;
      });
      motionTrailRef.current = motionTrailRef.current.filter(point => point.age < 1);

      landmarks.forEach(landmark => {
        if (landmark.visibility && landmark.visibility > 0.5) {
          motionTrailRef.current.push({
            x: landmark.x,
            y: landmark.y,
            age: 0,
          });
        }
      });

      drawVisualization(landmarks, worldLandmarks);
    } else {
      setHasPose(false);
      const canvas = overlayCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    }

    animationFrameRef.current = requestAnimationFrame(detectPose);
  }, [isCameraActive, visualMode]);

  const drawVisualization = (landmarks: Landmark[], worldLandmarks: Landmark[]) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (visualMode === "NEON") {
      drawNeonMode(ctx, landmarks, width, height);
    } else if (visualMode === "INK") {
      drawInkMode(ctx, landmarks, width, height);
    } else if (visualMode === "CYBER") {
      drawCyberMode(ctx, landmarks, worldLandmarks, width, height);
    } else if (visualMode === "ABSTRACT") {
      drawAbstractMode(ctx, landmarks, worldLandmarks, width, height);
    }
  };

  const drawNeonMode = (ctx: CanvasRenderingContext2D, landmarks: Landmark[], width: number, height: number) => {
    motionTrailRef.current.forEach(point => {
      const x = point.x * width;
      const y = point.y * height;
      const alpha = 1 - point.age;
      
      ctx.beginPath();
      ctx.arc(x, y, 3 * alpha, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(34, 211, 238, ${alpha * 0.8})`;
      ctx.shadowColor = "#22d3ee";
      ctx.shadowBlur = 20;
      ctx.fill();
    });

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 7],
      [0, 4], [4, 5], [5, 6],
      [5, 7], [7, 9], [9, 11], [11, 13], [13, 15],
      [6, 8], [8, 10], [10, 12], [12, 14], [14, 16],
      [5, 11], [6, 12],
      [11, 13], [13, 15],
      [12, 14], [14, 16],
    ];

    connections.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      if (p1 && p2 && p1.visibility && p2.visibility && p1.visibility > 0.5 && p2.visibility > 0.5) {
        const x1 = (1 - p1.x) * width;
        const y1 = p1.y * height;
        const x2 = (1 - p2.x) * width;
        const y2 = p2.y * height;

        const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
        gradient.addColorStop(0, "#22d3ee");
        gradient.addColorStop(1, "#a78bfa");

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 4;
        ctx.lineCap = "round";
        ctx.shadowColor = "#22d3ee";
        ctx.shadowBlur = 15;
        ctx.stroke();
      }
    });

    landmarks.forEach((landmark, i) => {
      if (landmark.visibility && landmark.visibility > 0.5) {
        const x = (1 - landmark.x) * width;
        const y = landmark.y * height;
        
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = "#22d3ee";
        ctx.shadowColor = "#22d3ee";
        ctx.shadowBlur = 20;
        ctx.fill();
      }
    });
  };

  const drawInkMode = (ctx: CanvasRenderingContext2D, landmarks: Landmark[], width: number, height: number) => {
    ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
    ctx.fillRect(0, 0, width, height);

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 7],
      [0, 4], [4, 5], [5, 6],
      [5, 7], [7, 9], [9, 11], [11, 13], [13, 15],
      [6, 8], [8, 10], [10, 12], [12, 14], [14, 16],
      [5, 11], [6, 12],
      [11, 13], [13, 15],
      [12, 14], [14, 16],
    ];

    connections.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      if (p1 && p2 && p1.visibility && p2.visibility && p1.visibility > 0.5 && p2.visibility > 0.5) {
        const x1 = (1 - p1.x) * width;
        const y1 = p1.y * height;
        const x2 = (1 - p2.x) * width;
        const y2 = p2.y * height;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        const midX = (x1 + x2) / 2 + (Math.random() - 0.5) * 5;
        const midY = (y1 + y2) / 2 + (Math.random() - 0.5) * 5;
        ctx.quadraticCurveTo(midX, midY, x2, y2);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.lineWidth = 3 + Math.random() * 2;
        ctx.lineCap = "round";
        ctx.stroke();
      }
    });
  };

  const drawCyberMode = (ctx: CanvasRenderingContext2D, landmarks: Landmark[], worldLandmarks: Landmark[], width: number, height: number) => {
    ctx.strokeStyle = "rgba(139, 92, 246, 0.2)";
    ctx.lineWidth = 1;
    for (let i = 0; i < width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
    }
    for (let i = 0; i < height; i += 40) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 7],
      [0, 4], [4, 5], [5, 6],
      [5, 7], [7, 9], [9, 11], [11, 13], [13, 15],
      [6, 8], [8, 10], [10, 12], [12, 14], [14, 16],
      [5, 11], [6, 12],
      [11, 13], [13, 15],
      [12, 14], [14, 16],
    ];

    connections.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      if (p1 && p2 && p1.visibility && p2.visibility && p1.visibility > 0.5 && p2.visibility > 0.5) {
        const x1 = (1 - p1.x) * width;
        const y1 = p1.y * height;
        const x2 = (1 - p2.x) * width;
        const y2 = p2.y * height;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = "#a78bfa";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.arc(x1, y1, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#a78bfa";
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(x1, y1, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();
      }
    });
  };

  const drawAbstractMode = (ctx: CanvasRenderingContext2D, landmarks: Landmark[], worldLandmarks: Landmark[], width: number, height: number) => {
    const nose = landmarks[0];
    if (!nose || !nose.visibility || nose.visibility < 0.5) return;

    const centerX = (1 - nose.x) * width;
    const centerY = nose.y * height;
    
    const avgVel = velocitiesRef.current.reduce((a, b) => a + b, 0) / velocitiesRef.current.length || 0;
    const intensity = Math.min(avgVel * 100, 1);

    const hue = (Date.now() / 50) % 360;
    
    for (let i = 0; i < 5; i++) {
      const radius = 50 + i * 30 + Math.sin(Date.now() / 500 + i) * 20;
      const alpha = 0.3 - i * 0.05;
      
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${alpha * intensity})`;
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    const leftHand = landmarks[9];
    const rightHand = landmarks[10];
    
    if (leftHand && leftHand.visibility && leftHand.visibility > 0.5) {
      const x = (1 - leftHand.x) * width;
      const y = leftHand.y * height;
      
      ctx.beginPath();
      ctx.arc(x, y, 20 * intensity, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${hue + 180}, 80%, 60%, 0.5)`;
      ctx.fill();
    }
    
    if (rightHand && rightHand.visibility && rightHand.visibility > 0.5) {
      const x = (1 - rightHand.x) * width;
      const y = rightHand.y * height;
      
      ctx.beginPath();
      ctx.arc(x, y, 20 * intensity, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${hue + 90}, 80%, 60%, 0.5)`;
      ctx.fill();
    }
  };

  const resetStats = () => {
    setMotionHistory([]);
    motionTrailRef.current = [];
    velocitiesRef.current = [];
    lastPositionRef.current = null;
    startTimeRef.current = Date.now();
    setStats({
      duration: 0,
      complexity: 0,
      avgVelocity: 0,
      peakVelocity: 0,
      directionChanges: 0,
      dominantMotion: "None",
    });
  };

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <header className="mb-6">
        <h1 className="text-4xl font-bold tracking-wider mb-2">
          <span className="text-cyan-400">MOTION</span>
          <span className="text-gray-500 mx-2">//</span>
          <span className="text-violet-400">DIRECTOR</span>
        </h1>
        <p className="text-sm text-gray-500 tracking-widest">HUMAN MOTION → GENERATIVE VISUALS</p>
      </header>

      {error && (
        <div className="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded text-red-400">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {/* Video with overlay */}
          <div className="aspect-video bg-gray-900/50 rounded-lg border border-white/10 relative overflow-hidden">
            {/* Live camera feed */}
            {showCamera && (
              <video
                ref={videoRef}
                className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]"
                playsInline
                muted
                autoPlay
              />
            )}
            
            {/* Overlay canvas for visualization */}
            <canvas
              ref={overlayCanvasRef}
              width={1280}
              height={720}
              className="absolute inset-0 w-full h-full"
            />
            
            {!isCameraActive && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-2xl text-gray-600 mb-2">CAMERA OFF</div>
                  <div className="text-sm text-gray-500">Click START CAMERA to begin</div>
                </div>
              </div>
            )}

            {isCameraActive && !hasPose && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <div className="text-center">
                  <div className="text-xl text-amber-400 mb-2 animate-pulse">SEARCHING FOR POSE</div>
                  <div className="text-sm text-gray-500">Position your full body in frame</div>
                </div>
              </div>
            )}
          </div>

          {/* Visual Mode Selector */}
          <div className="flex gap-3">
            {(["NEON", "INK", "CYBER", "ABSTRACT"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setVisualMode(mode)}
                className={`flex-1 py-3 px-4 rounded border transition-all ${
                  visualMode === mode
                    ? mode === "NEON" ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-400"
                    : mode === "INK" ? "bg-white/20 border-white/50 text-white"
                    : mode === "CYBER" ? "bg-violet-500/20 border-violet-500/50 text-violet-400"
                    : "bg-pink-500/20 border-pink-500/50 text-pink-400"
                    : "bg-white/5 border-white/10 text-gray-400 hover:border-white/30"
                }`}
              >
                {mode}
              </button>
            ))}
          </div>

          {/* Controls */}
          <div className="flex gap-3">
            {!isCameraActive ? (
              <button
                onClick={() => { startCamera(); resetStats(); }}
                className="flex-1 py-3 px-4 rounded border border-cyan-500/50 bg-cyan-500/20 text-cyan-400 hover:border-cyan-500 transition-all"
              >
                START CAMERA
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="flex-1 py-3 px-4 rounded border border-red-500/50 bg-red-500/20 text-red-400 hover:border-red-500 transition-all"
              >
                STOP CAMERA
              </button>
            )}
            <button
              onClick={() => setShowCamera(!showCamera)}
              className="flex-1 py-3 px-4 rounded border border-white/10 bg-white/5 text-gray-300 hover:border-white/30 transition-all"
            >
              {showCamera ? "HIDE CAMERA" : "SHOW CAMERA"}
            </button>
            <button
              onClick={resetStats}
              className="flex-1 py-3 px-4 rounded border border-white/10 bg-white/5 text-gray-300 hover:border-white/30 transition-all"
            >
              RESET
            </button>
          </div>
        </div>

        {/* Stats Panel */}
        <div className="space-y-4">
          <div className="p-4 bg-gray-900/50 rounded-lg border border-white/10">
            <h3 className="text-sm font-semibold text-gray-300 tracking-wider mb-4">PERFORMANCE ANALYSIS</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-xs text-gray-500">Duration</span>
                <span className="text-sm font-mono">{stats.duration.toFixed(2)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-500">Movement Complexity</span>
                <span className="text-sm font-mono">{stats.complexity.toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-500">Average Velocity</span>
                <span className="text-sm font-mono">{stats.avgVelocity.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-500">Peak Velocity</span>
                <span className="text-sm font-mono">{stats.peakVelocity.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-500">Direction Changes</span>
                <span className="text-sm font-mono">{stats.directionChanges}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-500">Dominant Motion</span>
                <span className="text-sm font-mono text-cyan-400">{stats.dominantMotion}</span>
              </div>
            </div>
          </div>

          <div className="p-4 bg-gray-900/50 rounded-lg border border-white/10">
            <h3 className="text-sm font-semibold text-gray-300 tracking-wider mb-4">MOTION TIMELINE</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              <AnimatePresence>
                {motionHistory.slice(-10).reverse().map((motion, i) => (
                  <motion.div
                    key={motion.timestamp}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center gap-3 p-2 bg-white/5 rounded text-xs"
                  >
                    <span className="text-gray-500 font-mono">
                      {((motion.timestamp - startTimeRef.current) / 1000).toFixed(1)}s
                    </span>
                    <span className="text-cyan-400">{motion.direction}</span>
                    <div className="flex-1 h-1 bg-gray-700 rounded overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-violet-500"
                        style={{ width: `${Math.min(100, motion.velocity * 100)}%` }}
                      />
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              {motionHistory.length === 0 && (
                <div className="text-center text-gray-600 text-sm py-8">No motion captured</div>
              )}
            </div>
          </div>

          <div className="p-4 bg-gray-900/50 rounded-lg border border-white/10">
            <h3 className="text-sm font-semibold text-gray-300 tracking-wider mb-4">POSE LANDMARKS</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
                "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
                "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
              ].map((landmark, i) => (
                <div key={landmark} className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${hasPose ? "bg-cyan-400" : "bg-gray-600"}`} />
                  <span className="text-gray-400">{landmark}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <footer className="mt-8 text-center text-xs text-gray-600">
        <p>MOTION//DIRECTOR — Human Motion to Generative Visuals</p>
        <p className="mt-1">Powered by MediaPipe Pose • Built with Next.js, React & Canvas</p>
      </footer>
    </div>
  );
}