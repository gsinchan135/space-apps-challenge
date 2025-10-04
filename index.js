const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

const logger = (req, res, next) => {
    console.log(`${req.method} ${req.url}`);
    next();
}

app.use(logger);

app.get('/', (req, res) => {
    res.send({ message: 'Hello, world!' });
});

app.get('/status', (req, res) => {
    res.send({ status: 'Server is running' });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});