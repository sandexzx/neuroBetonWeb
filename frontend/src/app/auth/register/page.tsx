"use client"

import { AuthForm } from "@/components/AuthForm";

export default function RegisterPage() {
  return (
    <div className="flex items-center justify-center py-8 bg-background">
      <AuthForm mode="register" />
    </div>
  );
}
