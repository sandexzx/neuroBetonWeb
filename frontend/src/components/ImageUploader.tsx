"use client"

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowUpTrayIcon } from "@heroicons/react/24/outline";
import { useAuth } from '@/lib/auth';

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
          {selectedFile && (
            <p className="text-sm text-muted-foreground">
              Selected file: {selectedFile.name}
            </p>
          )}
        </div>
        {prediction !== null && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Strength Prediction */}
            <div className="p-6 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-3 text-muted-foreground">Strength Prediction</h3>
              <p className="text-4xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">
                {prediction.strength.toFixed(2)} MPa
              </p>
            </div>

            {/* Cracks Detection */}
            <div className="p-6 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-3 text-muted-foreground">Cracks Detection</h3>
              <div className="space-y-2">
                <p className="text-2xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">
                  {prediction.cracks.has_cracks ? 'Cracks Detected' : 'No Cracks'}
                </p>
                <p className="text-sm text-muted-foreground">
                  Confidence: {(prediction.cracks.probability * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Concrete Type */}
            <div className="p-6 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-3 text-muted-foreground">Concrete Type</h3>
              <div className="space-y-2">
                <p className="text-2xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">
                  {prediction.concrete_type.type}
                </p>
                <p className="text-sm text-muted-foreground">
                  Confidence: {(prediction.concrete_type.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
