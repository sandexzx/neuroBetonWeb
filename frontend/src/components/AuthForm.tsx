"use client"

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { ArrowRightIcon, UserPlusIcon, EyeIcon, EyeSlashIcon } from "@heroicons/react/24/outline";
import { useAuth } from '@/lib/auth';
import { useToast } from '@/components/ui/toast';

const authSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  password: z.string().min(1, 'Password is required'),
});

interface AuthFormProps {
  mode: 'login' | 'register';
}

export function AuthForm({ mode }: AuthFormProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const router = useRouter();
  const { login, register: registerUser } = useAuth();
  const { addToast } = useToast();
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(authSchema),
    defaultValues: {
      username: '',
      password: '',
    },
  });

  const onSubmit = async (data: z.infer<typeof authSchema>) => {
    setIsLoading(true);
    try {
      if (mode === 'login') {
        await login(data.username, data.password);
        addToast({
          type: 'success',
          title: 'Login successful!',
          description: `Welcome back, ${data.username}!`
        });
      } else {
        await registerUser(data.username, data.password);
        addToast({
          type: 'success', 
          title: 'Account created!',
          description: 'You can now log in with your credentials.'
        });
        await login(data.username, data.password);
      }
      router.push('/dashboard');
    } catch (error: any) {
      console.error('Error:', error);
      addToast({
        type: 'error',
        title: mode === 'login' ? 'Login failed' : 'Registration failed',
        description: error.message || 'Please check your credentials and try again.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, ease: 'easeOut' }}
      className="w-full max-w-md mx-auto"
    >
      <Card className="bg-gray-100 dark:bg-gray-900 shadow-none border-0">
        <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl p-8 fade-in-up">
          <CardHeader className="text-center p-0 mb-6">
            <CardTitle className="text-4xl font-semibold tracking-tight mb-2 font-sans" style={{letterSpacing: '-0.02em'}}>
              {mode === 'login' ? 'Welcome Back' : 'Create Account'}
            </CardTitle>
            <CardDescription className="text-lg text-muted-foreground mb-2 font-sans">
              {mode === 'login' ? 'Sign in to your account' : 'Join NeuroBeton today'}
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="username" className="text-base">Username</Label>
                <Input
                  id="username"
                  type="text"
                  {...register('username', { required: true })}
                  disabled={isLoading}
                  className="h-11 text-base shadow-sm hover:shadow-md transition-shadow duration-200"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password" className="text-base">Password</Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    {...register('password', { required: true })}
                    disabled={isLoading}
                    className="h-11 text-base shadow-sm hover:shadow-md transition-shadow duration-200 pr-12"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
                  >
                    {showPassword ? (
                      <EyeSlashIcon className="w-5 h-5" />
                    ) : (
                      <EyeIcon className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>
              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-11 text-lg font-semibold shadow-md hover:scale-[1.03] active:scale-95 transition-transform duration-200 bg-black/80 dark:bg-white/10 hover:bg-black dark:hover:bg-white/20"
              >
                {isLoading ? (
                  'Loading...'
                ) : (
                  <>
                    {mode === 'login' ? (
                      <>
                        <ArrowRightIcon className="w-5 h-5 mr-2" /> Login
                      </>
                    ) : (
                      <>
                        <UserPlusIcon className="w-5 h-5 mr-2" /> Register
                      </>
                    )}
                  </>
                )}
              </Button>
              <div className="text-center text-sm text-muted-foreground">
                {mode === 'login' ? (
                  <p>
                    Don't have an account?{' '}
                    <button
                      type="button"
                      onClick={() => router.push('/auth/register')}
                      className="text-primary hover:underline"
                    >
                      Register
                    </button>
                  </p>
                ) : (
                  <p>
                    Already have an account?{' '}
                    <button
                      type="button"
                      onClick={() => router.push('/auth/login')}
                      className="text-primary hover:underline"
                    >
                      Login
                    </button>
                  </p>
                )}
              </div>
            </form>
          </CardContent>
        </div>
      </Card>
    </motion.div>
  );
}
