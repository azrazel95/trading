// alpacaClient.js
require('dotenv').config();
const Alpaca = require('@alpacahq/alpaca-trade-api');

const alpaca = new Alpaca({
  keyId:    process.env.APCA_API_KEY_ID,
  secretKey: process.env.APCA_API_SECRET_KEY,
  paper:    true,                             // still needed for paper trading
  baseUrl:  process.env.APCA_API_BASE_URL,    // <— make sure this is set
});

module.exports = alpaca;
