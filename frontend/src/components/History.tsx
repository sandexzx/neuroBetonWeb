"use client"

import { useEffect, useState } from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { motion } from 'framer-motion';
import { useAuth } from '@/lib/auth';

export function History() {
  const [history, setHistory] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { user } = useAuth();

  useEffect(() => {
    const fetchHistory = async () => {
      if (!user?.username) return;
      
      setIsLoading(true);
      try {
        console.log('Fetching history for user:', user.username);
        const response = await fetch(
          `http://localhost:8000/history/${user.username}`
        );
        const data = await response.json();
        console.log('Received data:', data);
        setHistory(data || []);
      } catch (error) {
        console.error('Error fetching history:', error);
        setHistory([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, [user]);

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
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date</TableHead>
            <TableHead>Image</TableHead>
            <TableHead>Strength</TableHead>
            <TableHead>Cracks</TableHead>
            <TableHead>Concrete Type</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {history.map((item, index) => (
            <TableRow key={index}>
              <TableCell>{new Date(item.date).toLocaleString()}</TableCell>
              <TableCell>
                <img 
                  src={item.image_url} 
                  alt="Concrete sample" 
                  className="w-20 h-20 object-cover rounded-lg"
                />
              </TableCell>
              <TableCell>{item.strength} MPa</TableCell>
              <TableCell>
                {item.cracks.has_cracks ? (
                  <span className="text-destructive">Yes ({Math.round(item.cracks.probability * 100)}%)</span>
                ) : (
                  <span className="text-green-600">No</span>
                )}
              </TableCell>
              <TableCell>
                {item.concrete_type.type} ({Math.round(item.concrete_type.confidence * 100)}%)
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
