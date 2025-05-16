// === server/index.js ===
const express = require('express');
const http = require('http');
const cors = require('cors');
const dotenv = require('dotenv');
const { Server } = require('socket.io');
const Alpaca = require('@alpacahq/alpaca-trade-api');

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*', methods: ['GET','POST'] } });

// Initialize Alpaca client
const alpaca = new Alpaca({
  keyId:    process.env.APCA_API_KEY_ID,
  secretKey: process.env.APCA_API_SECRET_KEY,
  paper:    true,
  baseUrl:  process.env.APCA_API_BASE_URL || 'https://paper-api.alpaca.markets'
});

const watchSymbol = 'SOXL';

// Seed endpoint: latest bid/ask
app.get('/api/quote-latest', async (req, res) => {
  try {
    const quote = await alpaca.getLatestQuote(watchSymbol);
    const bid = quote.BidPrice ?? quote.bp;
    const ask = quote.AskPrice ?? quote.ap;
    res.json({ bid, ask, timestamp: quote.Timestamp });
  } catch (err) {
    console.error('❌ /api/quote-latest error:', err.message);
    res.status(err.response?.status || 500).json({ error: err.message });
  }
});

// Account data
app.get('/api/account', async (req, res) => {
  try {
    const account = await alpaca.getAccount();
    res.json(account);
  } catch (err) {
    console.error('❌ /api/account error:', err.message);
    res.status(err.response?.status || 500).json({ error: err.message });
  }
});

// Portfolio positions
app.get('/api/portfolio', async (req, res) => {
  try {
    const positions = await alpaca.getPositions();
    res.json(positions);
  } catch (err) {
    console.error('❌ /api/portfolio error:', err.message);
    res.status(err.response?.status || 500).json({ error: err.message });
  }
});


// Execute trade endpoint
app.post('/api/trade', async (req, res) => {
  try {
    const { symbol, side, qty } = req.body;
    console.log(`💸 Placing ${side} order for ${qty} share(s) of ${symbol}`);
    const order = await alpaca.createOrder({
      symbol,
      qty,
      side,            // 'buy' or 'sell'
      type: 'market',
      time_in_force: 'day'
    });
    console.log('✅ Order response:', order);
    res.json(order);
  } catch (err) {
    console.error('❌ /api/trade error:', err);
    res.status(err.response?.status || 500).json({ error: err.message });
  }
});

// Real-time quote streaming
const alpacaStream = alpaca.data_stream_v2;
alpacaStream.onConnect(() => alpacaStream.subscribeForQuotes([watchSymbol]));
alpacaStream.onError(err => console.error('❌ Alpaca stream error:', err.message));
alpacaStream.onStockQuote(quote => {
  const bid = quote.BidPrice;
  const ask = quote.AskPrice;
  io.emit('stockQuote', { bid, ask, timestamp: quote.Timestamp });
});
alpacaStream.connect();

// Socket.IO connections
io.on('connection', socket => {
  console.log('Client connected');
  socket.on('disconnect', () => console.log('Client disconnected'));
});


server.listen(3001, () => console.log('Server listening on port 3001')); 



