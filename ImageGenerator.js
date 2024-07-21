import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Input, Button } from 'reactstrap';

const ImageGenerator = () => {
    const [prompt, setPrompt] = useState('');
    const [progress, setProgress] = useState('Waiting for prompt...');
    const [imageUrl, setImageUrl] = useState('');
    const [savePath, setSavePath] = useState('');
    const ws = useRef(null);

    useEffect(() => {
        ws.current = new WebSocket("ws://localhost:8000/ws");
        ws.current.binaryType = "arraybuffer";

        ws.current.onmessage = (event) => {
            if (typeof event.data === "string") {
                const data = JSON.parse(event.data);
                if (data.progress !== undefined) {
                    setProgress(`Progress: ${data.progress}%`);
                } else if (data.save_path) {
                    setSavePath(`Image saved at: ${data.save_path}`);
                }
            } else {
                setProgress('Image generated!');
                const blob = new Blob([event.data], { type: "image/png" });
                const url = URL.createObjectURL(blob);
                setImageUrl(url);
            }
        };

        return () => {
            ws.current.close();
        };
    }, []);

    const generateImage = () => {
        ws.current.send(JSON.stringify({ type: "prompt", prompt }));
        setProgress('Generation started...');
    };

    return (
        <Container>
            <Row className="mt-5">
                <Col>
                    <h1>Stable Diffusion Progress</h1>
                    <Input
                        type="text"
                        id="prompt-input"
                        placeholder="Enter your prompt here"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                    />
                    <Button color="primary" onClick={generateImage} className="mt-3">Generate Image</Button>
                    <p id="progress" className="mt-3">{progress}</p>
                    {imageUrl && <img id="generated-image" src={imageUrl} alt="Generated" style={{ display: 'block', marginTop: '20px' }} />}
                    <p id="save-path" className="mt-3">{savePath}</p>
                </Col>
            </Row>
        </Container>
    );
};

export default ImageGenerator;
