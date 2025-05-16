// === client/src/App.jsx ===
import { useEffect, useState } from 'react';
import axios from 'axios';
import { io } from 'socket.io-client';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';

export default function App() {
  const [account, setAccount]     = useState(null);
  const [portfolio, setPortfolio] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [latest, setLatest]       = useState(null);
  const [orderQty, setOrderQty]   = useState(1);
  const [orderSide, setOrderSide] = useState('buy');
 let watchSymbol = "SOXL"
  // Load account & portfolio
  useEffect(() => {
    axios.get('http://localhost:3001/api/account')
      .then(res => setAccount(res.data))
      .catch(err => console.error(err));
    axios.get('http://localhost:3001/api/portfolio')
      .then(res => setPortfolio(res.data))
      .catch(err => console.error(err));
  }, []);

  // Seed chart
  useEffect(() => {
    axios.get('http://localhost:3001/api/quote-latest')
      .then(res => {
        const mid = (res.data.bid + res.data.ask) / 2;
        setChartData([{ time: new Date(res.data.timestamp).toLocaleTimeString([], { hour12: false }), price: mid }]);
      })
      .catch(console.error);
  }, []);

  // Live updates
  useEffect(() => {
    const socket = io('http://localhost:3001');
    socket.on('stockQuote', data => setLatest((data.bid + data.ask) / 2));
    return () => socket.disconnect();
  }, []);
  useEffect(() => {
    const iv = setInterval(() => {
      if (latest !== null) {
        setChartData(prev => [
          ...prev.slice(-59),
          { time: new Date().toLocaleTimeString([], { hour12: false }), price: latest }
        ]);
      }
    }, 500);
    return () => clearInterval(iv);
  }, [latest]);

  // Execute trade
  const handleTrade = () => {
    axios.post('http://localhost:3001/api/trade', {
      symbol: watchSymbol,
      side: orderSide,
      qty: orderQty
    }).then(res => {
      alert(`Order ${res.data.side} ${res.data.qty} ${res.data.symbol} submitted!`);
    }).catch(err => {
      console.error(err);
      alert(`Trade error: ${err.response?.data.error || err.message}`);
    });
  };

return (
  <div className="p-6 font-sans bg-gray-50 min-h-screen">
    <h1 className="text-3xl font-extrabold mb-6 text-gray-800">📊 Portfolio & Live Bid/Ask Dashboard</h1>

    {/* Account & Portfolio */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      {account && (
        <div className="bg-white p-6 rounded-2xl shadow hover:shadow-lg">
          <h2 className="text-xl font-semibold mb-2">Account Info</h2>
          <p>Cash: ${account.cash}</p>
          <p>Equity: ${account.equity}</p>
          <p>Buying Power: ${account.buying_power}</p>
        </div>
      )}
      <div className="md:col-span-2 bg-white p-6 rounded-2xl shadow hover:shadow-lg">
        <h2 className="text-xl font-semibold mb-2">Current Positions</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-100">
              <tr>
                <th>Symbol</th><th>Qty</th><th>Avg Entry</th><th>Current</th><th>Mkt Value</th><th>P/L</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {portfolio.map(pos => (
                <tr key={pos.symbol} className="hover:bg-gray-50">
                  <td>{pos.symbol}</td>
                  <td>{pos.qty}</td>
                  <td>${pos.avg_entry_price}</td>
                  <td>${pos.current_price}</td>
                  <td>${pos.market_value}</td>
                  <td>${pos.unrealized_pl}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    {/* Trade Execution */}
    <div className="bg-white p-6 rounded-2xl shadow hover:shadow-lg mb-8">
      <h2 className="text-xl font-semibold mb-4">Execute Trade ({watchSymbol})</h2>
      <div className="flex items-center space-x-4">
        <select value={orderSide} onChange={e => setOrderSide(e.target.value)} className="border rounded px-2 py-1">
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
        <input type="number" min="1" value={orderQty} onChange={e => setOrderQty(Number(e.target.value))} className="border rounded px-2 py-1 w-20" />
        <button onClick={handleTrade} className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">Submit</button>
      </div>
    </div>

    {/* Live Quote Chart */}
    <div className="bg-white p-6 rounded-2xl shadow hover:shadow-lg">
      <h2 className="text-2xl mb-4">Live SOXL Mid-Price (0.5s intervals)</h2>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
          <XAxis dataKey="time" minTickGap={20} />
          <YAxis domain={['auto','auto']} />
          <Tooltip />
          <Area type="monotone" dataKey="price" stroke="#3b82f6" fill="url(#grad)" isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  </div>
);
}
