function StableDiffusion() {
  const [prompt, setPrompt] = useState('');
  const [progress, setProgress] = useState('Waiting for prompt...');
  const [imageSrc, setImageSrc] = useState('');
  const [savePath, setSavePath] = useState('');
  const [ws, setWs] = useState(null);

  useEffect(() => {
    const websocket = new WebSocket("ws://localhost:8000/ws");
    websocket.binaryType = "arraybuffer";
    setWs(websocket);

    websocket.onmessage = (event) => {
      if (typeof event.data === "string") {
        const data = JSON.parse(event.data);
        if (data.progress !== undefined) {
          setProgress(`Progress: ${data.progress}%`);
        } else if (data.save_path) {
          setSavePath(`Image saved at: ${data.save_path}`);
        }
      } else {
        setProgress('Image generated!');
        const blob = new Blob([event.data], {type: "image/png"});
        const url = URL.createObjectURL(blob);
        setImageSrc(url);
      }
    };

    return () => {
      websocket.close();
    };
  }, []);

  const generateImage = () => {
    if (ws) {
      ws.send(JSON.stringify({type: "prompt", prompt: prompt}));
      setProgress('Generation started...');
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
      <p>{progress}</p>
      {imageSrc && <img src={imageSrc} alt="Generated" style={{display: 'block', maxWidth: '100%'}} />}
      <p>{savePath}</p>
    </Container>
  );
}

export default StableDiffusion;
