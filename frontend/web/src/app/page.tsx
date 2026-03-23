"use client";
import { useState, useEffect } from "react";
import { Search, ShieldAlert, ShieldCheck, Activity, BarChart3, Clock, Globe } from "lucide-react";

const API_URL = "http://localhost:8000";

export default function Home() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    fetchHistory();
  }, []);

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_URL}/api/history`);
      if (res.ok) setHistory(await res.json());
    } catch (e) { console.error(e); }
  }

  async function analyze() {
    if (!url) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });
      const data = await res.json();
      setResult(data);
      fetchHistory();
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }

  return (
    <main className="min-h-screen bg-[#000000] text-gray-100 font-sans selection:bg-blue-500/30">
      {/* Background gradients iPhone style */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/30 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-900/20 blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto p-6 pt-16">
        <header className="mb-14 text-center space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-white/5 rounded-2xl border border-white/10 mb-4 backdrop-blur-xl">
             <Globe className="w-8 h-8 text-blue-400" />
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight bg-gradient-to-br from-white via-gray-200 to-gray-500 bg-clip-text text-transparent">
            Neural Truth Engine
          </h1>
          <p className="text-gray-400 text-lg max-w-xl mx-auto font-medium">
            Detecção avançada de Fake News no Bluesky orientada por Grafos e Votação Multimodelo (Ensemble).
          </p>
        </header>

        {/* Search Bar (Dynamic Island Style) */}
        <div className="max-w-3xl mx-auto bg-white/5 border border-white/10 p-2 rounded-[2rem] shadow-2xl backdrop-blur-xl flex items-center transition-all focus-within:ring-4 focus-within:ring-blue-500/20 focus-within:border-blue-500/40">
          <div className="pl-6 text-gray-400">
            <Search className="w-6 h-6" />
          </div>
          <input 
            type="text" 
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && analyze()}
            placeholder="Cole o link do post do Bluesky..."
            className="flex-1 bg-transparent border-none outline-none text-white px-6 py-4 text-lg placeholder-gray-500"
          />
          <button 
            onClick={analyze}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-full font-bold transition-all disabled:opacity-50 disabled:scale-95 active:scale-95 flex items-center gap-2"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : "Analisar"}
          </button>
        </div>

        {/* Main Result Area */}
        {result && (
          <div className="mt-12 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className={`p-1 rounded-3xl bg-gradient-to-b ${result.is_fake ? 'from-red-500/30 to-rose-900/10' : 'from-emerald-500/30 to-teal-900/10'}`}>
              <div className="bg-[#0a0f1a]/80 backdrop-blur-2xl p-8 rounded-[22px] border border-white/5 shadow-2xl relative overflow-hidden">
                
                <div className="flex flex-col md:flex-row items-center gap-8 z-10 relative">
                  <div className={`p-6 rounded-full flex-shrink-0 ${result.is_fake ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                    {result.is_fake ? <ShieldAlert className="w-20 h-20" /> : <ShieldCheck className="w-20 h-20" />}
                  </div>
                  
                  <div className="flex-1 text-center md:text-left">
                    <h2 className="text-3xl font-bold text-white mb-2">
                      {result.is_fake ? "Alerta de Fake News" : "Notícia Verificada"}
                    </h2>
                    <p className="text-gray-400 text-lg leading-relaxed line-clamp-3">
                      "{result.texto}"
                    </p>
                    <div className="mt-4 flex flex-wrap items-center justify-center md:justify-start gap-3">
                      <span className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-sm font-medium flex items-center gap-2">
                        <Activity className="w-4 h-4 text-blue-400" />
                        Confiança Ensemble: {(result.ensemble_score * 100).toFixed(1)}%
                      </span>
                      <span className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-sm font-medium flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-indigo-400" />
                        {result.nodes} Nós no Grafo
                      </span>
                    </div>
                  </div>
                </div>

                {/* Breakdown dos Modelos (Ensemble) */}
                <div className="mt-10 pt-8 border-t border-white/10">
                  <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-6 flex items-center gap-2">
                    Votação dos Modelos (Ablation)
                  </h3>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    {result.model_breakdown.map((m: any, idx: number) => (
                      <div key={idx} className="bg-white/5 p-5 rounded-2xl border border-white/5 hover:bg-white/10 transition-colors">
                        <div className="text-xs text-gray-500 mb-1 font-mono uppercase">Peso: {m.weight}x</div>
                        <div className="font-medium text-gray-200 mb-3 text-sm truncate" title={m.model.replace(/_/g, ' ')}>{m.model.replace(/_/g, ' ')}</div>
                        
                        <div className="flex items-end justify-between">
                           <span className={`text-2xl font-bold ${m.prob_fake >= 0.5 ? 'text-red-400' : 'text-emerald-400'}`}>
                             {(m.prob_fake * 100).toFixed(0)}%
                           </span>
                           <span className="text-xs text-gray-500 pb-1">Risco</span>
                        </div>
                        
                        <div className="mt-3 w-full bg-black/50 h-1.5 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full transition-all duration-1000 ${m.prob_fake >= 0.5 ? 'bg-red-500' : 'bg-emerald-500'}`}
                            style={{ width: `${m.prob_fake * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            </div>
          </div>
        )}

        {/* Historico */}
        {history.length > 0 && (
          <div className="mt-16 animate-in fade-in duration-1000 delay-300">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <Clock className="w-5 h-5 text-gray-400" /> Consultas Recentes
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {history.map(item => (
                <div key={item.id} className="bg-white/5 border border-white/10 p-5 rounded-3xl backdrop-blur-md hover:-translate-y-1 transition-transform cursor-pointer" onClick={() => {setUrl(item.url_bsky || ''); window.scrollTo(0,0);}}>
                  <div className="flex items-center gap-3 mb-3">
                    <div className={`w-3 h-3 rounded-full ${item.heuristica_final >= 0.5 ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]'}`} />
                    <span className="text-sm font-semibold text-gray-300 uppercase tracking-wider">{item.heuristica_final >= 0.5 ? 'Fake' : 'Real'}</span>
                  </div>
                  <p className="text-sm text-gray-400 line-clamp-2 leading-relaxed">"{item.texto_resumo}"</p>
                  <div className="mt-4 text-xs font-mono text-gray-600 truncate">{item.url_bsky ? item.url_bsky.replace('https://', '') : ''}</div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </main>
  );
}
