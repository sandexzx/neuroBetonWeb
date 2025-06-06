"use client"

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '@/lib/auth';
import { useToast } from '@/components/ui/toast';
import { CheckCircleIcon, XCircleIcon, CalendarIcon, PhotoIcon, BeakerIcon, CpuChipIcon } from "@heroicons/react/24/outline";

interface HistoryItem {
  filename: string;
  strength: number;
  cracks: {
    has_cracks: boolean;
    probability: number;
  };
  concrete_type: {
    type: string;
    confidence: number;
  };
  timestamp: string;
}

export function History() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { user } = useAuth();
  const { addToast } = useToast();

  useEffect(() => {
    const fetchHistory = async () => {
      if (!user?.username) return;
      
      setIsLoading(true);
      try {
        console.log('Fetching history for user:', user.username);
        const response = await fetch(
          `http://localhost:8000/history/${user.username}`
        );
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to load history');
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        setHistory(data || []);
      } catch (error: any) {
        console.error('Error fetching history:', error);
        addToast({
          type: 'error',
          title: 'Failed to load history',
          description: 'Unable to load your analysis history. Please try again.'
        });
        setHistory([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, [user, addToast]);

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('ru-RU', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    }).replace(',', '');
  };

  const formatFilename = (filename: string) => {
    // Remove timestamp prefix (format: YYYY-MM-DDThh:mm:ss.ssssss_)
    return filename.replace(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+_/, '');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-gray-200 border-t-gray-600 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-400">Загрузка истории...</p>
        </div>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <PhotoIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-400">История анализов пуста</p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
            Загрузите изображение для анализа прочности бетона
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto p-6">
      <div className="space-y-4">
        <AnimatePresence>
          {history.map((item, index) => {
            return (
              <motion.div
                key={`${item.timestamp}-${index}`}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                transition={{ 
                  duration: 0.4, 
                  delay: index * 0.05,
                  ease: [0.25, 0.46, 0.45, 0.94]
                }}
                whileHover={{ 
                  scale: 1.02,
                  y: -2,
                  transition: { duration: 0.2 }
                }}
                className="group relative"
              >
                <div className="bg-white/60 dark:bg-zinc-900/60 backdrop-blur-xl rounded-2xl p-6 shadow-lg border border-white/20 dark:border-white/10 hover:shadow-2xl transition-all duration-300">
                  {/* Header Row */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-500/10 rounded-lg">
                        <CalendarIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {formatDate(item.timestamp)}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white text-right">
                          {formatFilename(item.filename)}
                        </p>
                      </div>
                      <div className="p-2 bg-blue-500/10 rounded-lg">
                        <PhotoIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                      </div>
                    </div>
                  </div>

                  {/* Content Grid */}
                  <div className="space-y-4">
                    {/* Strength Analysis */}
                    <div className="flex items-center justify-between p-4 bg-gradient-to-r from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-xl backdrop-blur-sm">
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg bg-gradient-to-r ${item.strength >= 40 ? 'bg-emerald-500/10' : item.strength >= 30 ? 'bg-yellow-500/10' : 'bg-red-500/10'} bg-opacity-20`}>
                          <BeakerIcon className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                        </div>
                        <div>
                          <p className="font-medium text-gray-900 dark:text-white">
                            Прочность
                          </p>
                          <p className="text-sm text-muted-foreground max-w-[200px] truncate">
                            {item.strength.toFixed(2)} МПа
                          </p>
                        </div>
                      </div>
                    </div>
                      
                    {/* Cracks Detection */}
                    <div className="flex items-center justify-between p-4 bg-gradient-to-r from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-xl backdrop-blur-sm">
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg ${item.cracks.has_cracks ? 'bg-red-500/10' : 'bg-green-500/10'}`}>
                          {item.cracks.has_cracks ? (
                            <XCircleIcon className="w-5 h-5 text-red-600 dark:text-red-400" />
                          ) : (
                            <CheckCircleIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium text-gray-900 dark:text-white">
                            Трещины
                          </p>
                          <p className="text-sm text-muted-foreground max-w-[200px] truncate">
                            {item.cracks.has_cracks ? 'Обнаружены' : 'Не обнаружены'}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-bold text-red-600 dark:text-red-400">
                          {Math.round(item.cracks.probability * 100)}%
                        </p>
                        <p className="text-xs text-muted-foreground">уверенность</p>
                      </div>
                    </div>

                    {/* Concrete Type */}
                    <div className="flex items-center justify-between p-4 bg-gradient-to-r from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-xl backdrop-blur-sm">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-purple-500/10 rounded-lg">
                          <CpuChipIcon className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                        </div>
                        <div>
                          <p className="font-medium text-gray-900 dark:text-white">
                            Тип материала
                          </p>
                          <p className="text-sm text-muted-foreground max-w-[200px] truncate">
                            {item.concrete_type.type}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                          {Math.round(item.concrete_type.confidence * 100)}%
                        </p>
                        <p className="text-xs text-muted-foreground">уверенность</p>
                      </div>
                    </div>
                  </div>

                  {/* Hover Effect Gradient */}
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 to-purple-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}