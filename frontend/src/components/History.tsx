"use client"

import { useEffect, useState } from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { motion } from 'framer-motion';
import { useAuth } from '@/lib/auth';
import { useToast } from '@/components/ui/toast';

export function History() {
  const [history, setHistory] = useState<any[]>([]);
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

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <p className="text-center text-muted-foreground">Loading history...</p>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <p className="text-center text-muted-foreground">No history available</p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto p-6">
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
              <TableCell>{new Date(item.timestamp).toLocaleString('ru-RU', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              })}</TableCell>
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
