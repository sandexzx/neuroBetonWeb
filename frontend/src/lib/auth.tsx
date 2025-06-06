"use client"

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  username: string;
}

interface AuthContextType {
  user: User | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  register: (username: string, password: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    const initializeAuth = () => {
      try {
        const savedUser = localStorage.getItem('user');
        console.log('AuthProvider - Initial localStorage user:', savedUser);
        
        if (savedUser) {
          const parsedUser = JSON.parse(savedUser);
          console.log('AuthProvider - Parsed user from localStorage:', parsedUser);
          
          if (parsedUser && parsedUser.username) {
            console.log('AuthProvider - Setting user from localStorage:', parsedUser);
            setUser(parsedUser);
          } else {
            console.error('AuthProvider - Invalid user data in localStorage');
            localStorage.removeItem('user');
          }
        } else {
          console.log('AuthProvider - No user found in localStorage');
        }
      } catch (error) {
        console.error('AuthProvider - Error initializing auth:', error);
        localStorage.removeItem('user');
      } finally {
        setIsInitialized(true);
      }
    };

    initializeAuth();
  }, []);

  const login = async (username: string, password: string) => {
    try {
      console.log('AuthProvider - Attempting login for:', username);
      const response = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || 'Login failed';
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log('AuthProvider - Login response:', data);
      
      if (!data.username) {
        throw new Error('Server did not return username');
      }

      const user = { username: data.username };
      console.log('AuthProvider - Setting user after login:', user);
      setUser(user);
      localStorage.setItem('user', JSON.stringify(user));
    } catch (error) {
      console.error('AuthProvider - Login error:', error);
      throw error;
    }
  };

  const register = async (username: string, password: string) => {
    try {
      console.log('AuthProvider - Attempting registration for:', username);
      const response = await fetch('http://localhost:8000/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Registration failed');
      }
      console.log('AuthProvider - Registration successful for:', username);
    } catch (error) {
      console.error('AuthProvider - Registration error:', error);
      throw error;
    }
  };

  const logout = () => {
    console.log('AuthProvider - Logging out user:', user);
    setUser(null);
    localStorage.removeItem('user');
  };

  if (!isInitialized) {
    console.log('AuthProvider - Not yet initialized');
    return null;
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, register }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
} 