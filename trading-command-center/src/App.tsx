import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Orders from './pages/Orders';
import Strategies from './pages/Strategies';
import Brokers from './pages/Brokers';
import Risk from './pages/Risk';
import AIAssistant from './pages/AIAssistant';
import Configuration from './pages/Configuration';
import Analytics from './pages/Analytics';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-matrix-black text-matrix-green font-mono">
          <Toaster position="top-right" />
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/orders" element={<Orders />} />
              <Route path="/strategies" element={<Strategies />} />
              <Route path="/brokers" element={<Brokers />} />
              <Route path="/risk" element={<Risk />} />
              <Route path="/ai" element={<AIAssistant />} />
              <Route path="/analytics/*" element={<Analytics />} />
              <Route path="/config" element={<Configuration />} />
            </Routes>
          </Layout>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
