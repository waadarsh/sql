import React, { useState, useEffect, useRef } from 'react';

const StableDiffusionProgress = () => {
  const [prompt, setPrompt] = useState('');
  const [progress, setProgress] = useState('Waiting for prompt...');
  const [imageSrc, setImageSrc] = useState('');
  const [savePath, setSavePath] = useState('');
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws');
    ws.current.binaryType = 'arraybuffer';

    ws.current.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const data = JSON.parse(event.data);
        if (data.progress !== undefined) {
          setProgress(`Progress: ${data.progress}%`);
        } else if (data.save_path) {
          setSavePath(`Image saved at: ${data.save_path}`);
        }
      } else {
        setProgress('Image generated!');
        const blob = new Blob([event.data], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        setImageSrc(url);
      }
    };

    return () => {
      ws.current.close();
    };
  }, []);

  const generateImage = () => {
    ws.current.send(JSON.stringify({ type: 'prompt', prompt }));
    setProgress('Generation started...');
  };

  return (
    <div>
      <h1>Stable Diffusion Progress</h1>
      <input
        type="text"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter your prompt here"
      />
      <button onClick={generateImage}>Generate Image</button>
      <p>{progress}</p>
      {imageSrc && <img src={imageSrc} alt="Generated" style={{ display: 'block' }} />}
      <p>{savePath}</p>
    </div>
  );
};

export default StableDiffusionProgress;
