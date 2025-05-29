import type React from "react"
import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "NeuralPaze",
  description: "Design Your Neural Network",
  generator: "TheVixhal",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Immediate ResizeObserver suppression
              (function() {
                const originalError = console.error;
                console.error = function(...args) {
                  if (args[0] && args[0].toString().includes('ResizeObserver')) {
                    return;
                  }
                  originalError.apply(console, args);
                };
                
                window.addEventListener('error', function(e) {
                  if (e.message && e.message.includes('ResizeObserver')) {
                    e.stopImmediatePropagation();
                    e.preventDefault();
                    return false;
                  }
                }, true);
              })();
            `,
          }}
        />
      </head>
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}
