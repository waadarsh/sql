import React from 'react';
import { Container, Row, Col } from 'reactstrap';
import ImageGenerator from './ImageGenerator';

const App = () => {
    return (
        <Container>
            <Row className="mt-5">
                <Col>
                    <ImageGenerator />
                </Col>
            </Row>
        </Container>
    );
};

export default App;
