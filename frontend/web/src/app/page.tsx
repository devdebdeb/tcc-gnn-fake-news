"use client";
import { useState, useEffect } from "react";

const API_URL = "http://localhost:8000";

export default function Home() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => { fetchHistory(); }, []);

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_URL}/api/history`);
      if (res.ok) setHistory(await res.json());
    } catch {}
  }

  return (
    <main className="min-h-screen bg-[#050b14] text-white p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 text-center">
          <h1 className="text-5xl font-extrabold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
            Truth GNN Analytics
          </h1>
          <p className="text-slate-400 mt-2">Portal de Detecção de Fake News via Grafos (Restaurado)</p>
        </header>

        <div className="bg-white/5 border border-white/10 p-8 rounded-3xl backdrop-blur-xl">
          <input 
            type="text" 
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Cole o link do post do Bluesky..."
            className="w-full bg-black/40 border border-white/10 p-4 rounded-xl mb-4"
          />
          <button className="w-full bg-blue-600 p-4 rounded-xl font-bold hover:bg-blue-700 transition-all">
            Analisar Agora
          </button>
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
            <h2 className="text-sm uppercase text-slate-500 mb-2">Histórico Recente</h2>
            <div className="space-y-3">
              {history.map(item => (
                <div key={item.id} className="text-sm p-3 bg-black/20 rounded-lg">
                  {item.texto_resumo}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
