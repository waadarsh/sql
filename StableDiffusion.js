// import React, { useState, useEffect } from 'react';
// import { Container, Input, Button, Progress, Alert } from 'reactstrap';
// import 'bootstrap/dist/css/bootstrap.min.css';

// function StableDiffusion() {
//   const [prompt, setPrompt] = useState('');
//   const [progress, setProgress] = useState(0);
//   const [progressText, setProgressText] = useState('Waiting for prompt...');
//   const [imageSrc, setImageSrc] = useState('');
//   const [savePath, setSavePath] = useState('');
//   const [ws, setWs] = useState(null);
//   const [error, setError] = useState('');

//   useEffect(() => {
//     const websocket = new WebSocket("ws://localhost:8000/ws");
//     websocket.binaryType = "arraybuffer";

//     websocket.onopen = () => {
//       setWs(websocket);
//     };

//     websocket.onmessage = (event) => {
//       if (typeof event.data === "string") {
//         const data = JSON.parse(event.data);
//         if (data.progress !== undefined) {
//           setProgress(data.progress);
//           setProgressText(`Progress: ${data.progress}%`);
//         } else if (data.save_path) {
//           setSavePath(`Image saved at: ${data.save_path}`);
//         }
//       } else {
//         setProgress(100);
//         setProgressText('Image generated!');
//         const blob = new Blob([event.data], { type: "image/png" });
//         const url = URL.createObjectURL(blob);
//         setImageSrc(url);
//       }
//     };

//     websocket.onerror = (error) => {
//       setError('WebSocket error: ' + error.message);
//     };

//     websocket.onclose = () => {
//       setWs(null);
//       setProgressText('Connection closed');
//     };

//     return () => {
//       websocket.close();
//     };
//   }, []);

//   const generateImage = () => {
//     if (ws && prompt.trim()) {
//       ws.send(JSON.stringify({ type: "prompt", prompt: prompt }));
//       setProgress(0);
//       setProgressText('Generation started...');
//       setImageSrc('');
//       setSavePath('');
//       setError('');
//     } else {
//       setError('Please enter a valid prompt.');
//     }
//   };

//   return (
//     <Container>
//       <h1>Stable Diffusion Progress</h1>
//       <Input
//         type="text"
//         value={prompt}
//         onChange={(e) => setPrompt(e.target.value)}
//         placeholder="Enter your prompt here"
//       />
//       <Button color="primary" onClick={generateImage}>Generate Image</Button>
//       <Progress value={progress} className="my-3" />
//       <p>{progressText}</p>
//       {imageSrc && <img src={imageSrc} alt="Generated" style={{ display: 'block', maxWidth: '100%' }} />}
//       <p>{savePath}</p>
//       {error && <Alert color="danger">{error}</Alert>}
//     </Container>
//   );
// }

// export default StableDiffusion;


import React, { useState, useEffect, useRef } from 'react';
import { Container, Input, Button, Progress, FormGroup, Label } from 'reactstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

function StableDiffusion() {
  const [prompt, setPrompt] = useState('');
  const [progress, setProgress] = useState('Waiting for prompt...');
  const [imageSrc, setImageSrc] = useState('');
  const [savePath, setSavePath] = useState('');
  const [ws, setWs] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const fileInputRef = useRef(null);

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
      const data = {
        type: "prompt",
        prompt: prompt
      };

      if (uploadedImage) {
        const reader = new FileReader();
        reader.onload = (e) => {
          data.image = e.target.result;
          ws.send(JSON.stringify(data));
        };
        reader.readAsDataURL(uploadedImage);
      } else {
        ws.send(JSON.stringify(data));
      }

      setProgress('Generation started...');
    }
  };

  const handleImageUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedImage(e.target.files[0]);
    }
  };

  const clearUploadedImage = () => {
    setUploadedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Container>
      <h1>Stable Diffusion Progress</h1>
      <FormGroup>
        <Label for="prompt-input">Prompt</Label>
        <Input 
          id="prompt-input"
          type="text" 
          value={prompt} 
          onChange={(e) => setPrompt(e.target.value)} 
          placeholder="Enter your prompt here"
        />
      </FormGroup>
      <FormGroup>
        <Label for="image-upload">Upload Image (Optional)</Label>
        <Input 
          id="image-upload"
          type="file" 
          onChange={handleImageUpload} 
          accept="image/*"
          innerRef={fileInputRef}
        />
      </FormGroup>
      {uploadedImage && (
        <div>
          <p>Image uploaded: {uploadedImage.name}</p>
          <Button color="secondary" onClick={clearUploadedImage}>Clear Image</Button>
        </div>
      )}
      <Button color="primary" onClick={generateImage} className="mt-3">Generate Image</Button>
      <p className="mt-3">{progress}</p>
      {imageSrc && <img src={imageSrc} alt="Generated" style={{display: 'block', maxWidth: '100%'}} />}
      <p>{savePath}</p>
    </Container>
  );
}

export default StableDiffusion;
