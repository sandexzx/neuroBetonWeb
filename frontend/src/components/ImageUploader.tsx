"use client"

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowUpTrayIcon } from "@heroicons/react/24/outline";
import { useAuth } from '@/lib/auth';
import { motion } from 'framer-motion';

interface PredictionResult {
  strength: number;
  cracks: {
    has_cracks: boolean;
    probability: number;
  };
  concrete_type: {
    type: string;
    confidence: number;
  };
}

export function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const { user } = useAuth();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setPrediction(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile || !user) return;

    setLoading(true);
    setPrediction(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`http://localhost:8000/predict?username=${user.username}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze image');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full">
      <div className="space-y-8">
        <div className="flex flex-col space-y-4">
          <div className="relative">
            <div className="flex items-center gap-4">
              <div className="relative flex-1">
                <Input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  disabled={loading}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="flex items-center justify-center h-10 px-6 text-base font-semibold shadow-md hover:scale-[1.02] active:scale-95 transition-all duration-200 bg-black/80 dark:bg-white/10 hover:bg-black dark:hover:bg-white/20 rounded-lg cursor-pointer"
                >
                  {selectedFile ? selectedFile.name : 'Choose File'}
                </label>
              </div>
              <Button
                onClick={handleSubmit}
                disabled={!selectedFile || loading || !user}
                className="h-10 px-6 text-base font-semibold shadow-md hover:scale-[1.02] active:scale-95 transition-all duration-200 bg-black/80 dark:bg-white/10 hover:bg-black dark:hover:bg-white/20 rounded-lg"
              >
                {loading ? 'Processing...' : (
                  <>
                    <ArrowUpTrayIcon className="w-5 h-5 mr-2" />
                    Analyze
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
        {selectedFile && (
          <p className="text-sm text-muted-foreground">
            Selected file: {selectedFile.name}
          </p>
        )}
        {prediction !== null && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 30 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ 
              duration: 0.6, 
              ease: [0.25, 0.46, 0.45, 0.94],
              staggerChildren: 0.2
            }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {/* Strength Prediction */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2, duration: 0.4 }}
              className="p-8 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm border border-white/20 dark:border-white/10 flex flex-col items-center"
            >
              <motion.h3 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2, duration: 0.4 }}
                className="text-lg font-semibold mb-4 text-muted-foreground text-center w-full"
              >
                Analysis Complete
              </motion.h3>
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ 
                  delay: 0.4, 
                  duration: 0.5, 
                  type: "spring", 
                  stiffness: 200 
                }}
                className="text-center"
              >
                <motion.span 
                  className="text-5xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500"
                  initial={{ backgroundPosition: "0% 50%" }}
                  animate={{ backgroundPosition: "100% 50%" }}
                  transition={{ 
                    duration: 2, 
                    repeat: Infinity, 
                    repeatType: "reverse",
                    ease: "easeInOut"
                  }}
                  style={{ backgroundSize: "200% 200%" }}
                >
                  {prediction.strength.toFixed(2)}
                </motion.span>
                <motion.span 
                  className="text-2xl font-semibold text-muted-foreground ml-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8, duration: 0.3 }}
                >
                  MPa
                </motion.span>
              </motion.div>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ delay: 0.6, duration: 0.8, ease: "easeOut" }}
                className="h-1 bg-gradient-to-r from-emerald-500 via-blue-500 to-purple-500 rounded-full mt-4 mx-auto"
              />
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0, duration: 0.4 }}
                className="text-center text-sm text-muted-foreground mt-3"
              >
                Concrete strength prediction
              </motion.p>
            </motion.div>

            {/* Cracks Detection */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.4 }}
              className="p-8 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm border border-white/20 dark:border-white/10 flex flex-col items-center"
            >
              <motion.h3 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.4 }}
                className="text-lg font-semibold mb-4 text-muted-foreground text-center w-full"
              >
                Cracks Detection
              </motion.h3>
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5, duration: 0.5 }}
                className="text-center"
              >
                <motion.p 
                  className="text-2xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6, duration: 0.4 }}
                >
                  {prediction.cracks.has_cracks ? 'Cracks Detected' : 'No Cracks'}
                </motion.p>
                <motion.p 
                  className="text-sm text-muted-foreground mt-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.7, duration: 0.4 }}
                >
                  Confidence: {(prediction.cracks.probability * 100).toFixed(1)}%
                </motion.p>
              </motion.div>
            </motion.div>

            {/* Concrete Type */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.4 }}
              className="p-8 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm border border-white/20 dark:border-white/10 flex flex-col items-center"
            >
              <motion.h3 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.4 }}
                className="text-lg font-semibold mb-4 text-muted-foreground text-center w-full"
              >
                Concrete Type
              </motion.h3>
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6, duration: 0.5 }}
                className="text-center"
              >
                <motion.p 
                  className="text-2xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.7, duration: 0.4 }}
                >
                  {prediction.concrete_type.type}
                </motion.p>
                <motion.p 
                  className="text-sm text-muted-foreground mt-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8, duration: 0.4 }}
                >
                  Confidence: {(prediction.concrete_type.confidence * 100).toFixed(1)}%
                </motion.p>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
