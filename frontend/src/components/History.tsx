"use client"

import { useEffect, useState } from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { motion } from 'framer-motion';

export function History() {
  const [history, setHistory] = useState<any[]>([]);
  const [username, setUsername] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setUsername(localStorage.getItem('username') || '');
  }, []);

  useEffect(() => {
    const fetchHistory = async () => {
      if (!username) return;
      
      setIsLoading(true);
      try {
        const response = await fetch(
          `http://localhost:8000/api/history?username=${username}`
        );
        const data = await response.json();
        setHistory(data.history || []);
      } catch (error) {
        console.error('Error fetching history:', error);
        setHistory([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, [username]);

  if (isLoading) {
    return (
      <div className="p-8 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm">
        <p className="text-center text-muted-foreground">Loading history...</p>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="p-8 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm">
        <p className="text-center text-muted-foreground">No history available</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="overflow-hidden rounded-xl bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 backdrop-blur-sm"
    >
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="border-b border-black/10 dark:border-white/10">
              <TableHead className="text-base font-semibold text-muted-foreground">Filename</TableHead>
              <TableHead className="text-base font-semibold text-muted-foreground">Strength Prediction</TableHead>
              <TableHead className="text-base font-semibold text-muted-foreground">Timestamp</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.map((item, index) => (
              <TableRow 
                key={index}
                className="border-b border-black/10 dark:border-white/10 hover:bg-black/5 dark:hover:bg-white/5 transition-colors duration-200"
              >
                <TableCell className="text-base">{item.filename}</TableCell>
                <TableCell className="text-base">
                  <span className="font-semibold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">
                    {item.strength_prediction} MPa
                  </span>
                </TableCell>
                <TableCell className="text-base text-muted-foreground">
                  {new Date(item.timestamp).toLocaleString()}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </motion.div>
  );
}
