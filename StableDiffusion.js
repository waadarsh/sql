import React, { useState, useEffect } from 'react';
import { Container, Input, Button, Progress, Alert } from 'reactstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

function StableDiffusion() {
  const [prompt, setPrompt] = useState('');
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('Waiting for prompt...');
  const [imageSrc, setImageSrc] = useState('');
  const [savePath, setSavePath] = useState('');
  const [ws, setWs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const websocket = new WebSocket("ws://localhost:8000/ws");
    websocket.binaryType = "arraybuffer";

    websocket.onopen = () => {
      setWs(websocket);
    };

    websocket.onmessage = (event) => {
      if (typeof event.data === "string") {
        const data = JSON.parse(event.data);
        if (data.progress !== undefined) {
          setProgress(data.progress);
          setProgressText(`Progress: ${data.progress}%`);
        } else if (data.save_path) {
          setSavePath(`Image saved at: ${data.save_path}`);
        }
      } else {
        setProgress(100);
        setProgressText('Image generated!');
        const blob = new Blob([event.data], { type: "image/png" });
        const url = URL.createObjectURL(blob);
        setImageSrc(url);
      }
    };

    websocket.onerror = (error) => {
      setError('WebSocket error: ' + error.message);
    };

    websocket.onclose = () => {
      setWs(null);
      setProgressText('Connection closed');
    };

    return () => {
      websocket.close();
    };
  }, []);

  const generateImage = () => {
    if (ws && prompt.trim()) {
      ws.send(JSON.stringify({ type: "prompt", prompt: prompt }));
      setProgress(0);
      setProgressText('Generation started...');
      setImageSrc('');
      setSavePath('');
      setError('');
    } else {
      setError('Please enter a valid prompt.');
    }
  };

  return (
    <Container>
      <h1>Stable Diffusion Progress</h1>
      <Input
        type="text"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter your prompt here"
      />
      <Button color="primary" onClick={generateImage}>Generate Image</Button>
      <Progress value={progress} className="my-3" />
      <p>{progressText}</p>
      {imageSrc && <img src={imageSrc} alt="Generated" style={{ display: 'block', maxWidth: '100%' }} />}
      <p>{savePath}</p>
      {error && <Alert color="danger">{error}</Alert>}
    </Container>
  );
}

export default StableDiffusion;
