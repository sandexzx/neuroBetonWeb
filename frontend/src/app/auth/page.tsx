import { AuthForm } from '@/components/AuthForm';

export default function AuthPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-md">
        <AuthForm mode="login" />
      </div>
    </div>
  );
}
