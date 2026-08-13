import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Toaster } from '@/components/ui/toaster';
import {
  KeyGenerationComponent,
  EncryptionComponent,
  DecryptionComponent,
  SigningComponent,
  VerificationComponent,
  KeyManagementComponent
} from '@/components/RSAComponents';
import { RSAKeyPair } from '@/lib/rsa';
import { Shield, Key, Lock, Unlock, FileKey, CheckCircle, Settings } from 'lucide-react';

const Index = () => {
  const [keyPair, setKeyPair] = useState<RSAKeyPair | null>(null);

  const handleKeyGenerated = (newKeyPair: RSAKeyPair) => {
    setKeyPair(newKeyPair);
  };

  const handleKeyImported = (importedKeys: Partial<RSAKeyPair>) => {
    setKeyPair(prev => ({
      publicKey: importedKeys.publicKey || prev?.publicKey || null,
      privateKey: importedKeys.privateKey || prev?.privateKey || null
    }) as RSAKeyPair);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Shield className="h-8 w-8 text-primary" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              RSA Cryptography Tool
            </h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            A comprehensive RSA encryption, decryption, and digital signature tool with advanced security features
          </p>
          
          {/* Security Features */}
          <div className="flex flex-wrap justify-center gap-2 mt-4">
            <Badge variant="secondary">2048/3072/4096-bit Keys</Badge>
            <Badge variant="secondary">OAEP Padding</Badge>
            <Badge variant="secondary">Digital Signatures</Badge>
            <Badge variant="secondary">Secure Random Generation</Badge>
            <Badge variant="secondary">Key Import/Export</Badge>
          </div>
        </div>

        {/* Key Status */}
        {keyPair && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                Current Key Pair Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-2">
                  <Badge variant={keyPair.publicKey ? "default" : "secondary"}>
                    {keyPair.publicKey ? "✅ Public Key Available" : "❌ No Public Key"}
                  </Badge>
                  {keyPair.publicKey && (
                    <span className="text-sm text-muted-foreground">
                      {keyPair.publicKey.keySize} bits
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={keyPair.privateKey ? "default" : "secondary"}>
                    {keyPair.privateKey ? "✅ Private Key Available" : "❌ No Private Key"}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Main Tabs */}
        <Tabs defaultValue="keygen" className="space-y-6">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="keygen" className="flex items-center gap-2">
              <Key className="h-4 w-4" />
              <span className="hidden sm:inline">Key Gen</span>
            </TabsTrigger>
            <TabsTrigger value="encrypt" className="flex items-center gap-2">
              <Lock className="h-4 w-4" />
              <span className="hidden sm:inline">Encrypt</span>
            </TabsTrigger>
            <TabsTrigger value="decrypt" className="flex items-center gap-2">
              <Unlock className="h-4 w-4" />
              <span className="hidden sm:inline">Decrypt</span>
            </TabsTrigger>
            <TabsTrigger value="sign" className="flex items-center gap-2">
              <FileKey className="h-4 w-4" />
              <span className="hidden sm:inline">Sign</span>
            </TabsTrigger>
            <TabsTrigger value="verify" className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              <span className="hidden sm:inline">Verify</span>
            </TabsTrigger>
            <TabsTrigger value="manage" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              <span className="hidden sm:inline">Manage</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="keygen" className="space-y-6">
            <KeyGenerationComponent onKeyGenerated={handleKeyGenerated} />
            
            <Alert>
              <Shield className="h-4 w-4" />
              <AlertDescription>
                <strong>Security Features:</strong>
                <ul className="mt-2 space-y-1 text-sm">
                  <li>• Cryptographically secure pseudo-random number generator (CSPRNG)</li>
                  <li>• Miller-Rabin primality testing for strong prime generation</li>
                  <li>• Support for 2048, 3072, and 4096-bit key lengths</li>
                  <li>• Extended Euclidean algorithm for modular inverse computation</li>
                </ul>
              </AlertDescription>
            </Alert>
          </TabsContent>

          <TabsContent value="encrypt" className="space-y-6">
            <EncryptionComponent publicKey={keyPair?.publicKey || null} />
            
            <Alert>
              <Lock className="h-4 w-4" />
              <AlertDescription>
                <strong>Encryption Features:</strong>
                <ul className="mt-2 space-y-1 text-sm">
                  <li>• OAEP padding for enhanced security against chosen-ciphertext attacks</li>
                  <li>• PKCS#1 v1.5 padding for compatibility with legacy systems</li>
                  <li>• Support for various input formats (text, files)</li>
                  <li>• Hexadecimal output format for easy transmission</li>
                </ul>
              </AlertDescription>
            </Alert>
          </TabsContent>

          <TabsContent value="decrypt" className="space-y-6">
            <DecryptionComponent privateKey={keyPair?.privateKey || null} />
            
            <Alert>
              <Unlock className="h-4 w-4" />
              <AlertDescription>
                <strong>Decryption Features:</strong>
                <ul className="mt-2 space-y-1 text-sm">
                  <li>• Automatic padding scheme detection and removal</li>
                  <li>• Support for both OAEP and PKCS#1 v1.5 padded ciphertexts</li>
                  <li>• Secure private key operations with constant-time algorithms</li>
                  <li>• Error handling for invalid ciphertexts and padding</li>
                </ul>
              </AlertDescription>
            </Alert>
          </TabsContent>

          <TabsContent value="sign" className="space-y-6">
            <SigningComponent privateKey={keyPair?.privateKey || null} />
            
            <Alert>
              <FileKey className="h-4 w-4" />
              <AlertDescription>
                <strong>Digital Signing Features:</strong>
                <ul className="mt-2 space-y-1 text-sm">
                  <li>• RSA-based digital signatures for message authentication</li>
                  <li>• Hash-based message integrity verification</li>
                  <li>• Support for various message formats and sizes</li>
                  <li>• Exportable signature packages for easy sharing</li>
                </ul>
              </AlertDescription>
            </Alert>
          </TabsContent>

          <TabsContent value="verify" className="space-y-6">
            <VerificationComponent publicKey={keyPair?.publicKey || null} />
            
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>
                <strong>Signature Verification Features:</strong>
                <ul className="mt-2 space-y-1 text-sm">
                  <li>• Public key signature verification for authenticity</li>
                  <li>• Message integrity checking against tampering</li>
                  <li>• Support for both individual signatures and JSON packages</li>
                  <li>• Clear verification status with detailed feedback</li>
                </ul>
              </AlertDescription>
            </Alert>
          </TabsContent>

          <TabsContent value="manage" className="space-y-6">
            <KeyManagementComponent 
              keyPair={keyPair} 
              onKeyImported={handleKeyImported} 
            />
            
            <Alert>
              <Settings className="h-4 w-4" />
              <AlertDescription>
                <strong>Key Management Features:</strong>
                <ul className="mt-2 space-y-1 text-sm">
                  <li>• Secure key export in JSON format for backup and sharing</li>
                  <li>• Key import functionality for existing key pairs</li>
                  <li>• Separate handling of public and private keys</li>
                  <li>• Key rotation support for enhanced security</li>
                </ul>
              </AlertDescription>
            </Alert>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="mt-12 text-center text-sm text-muted-foreground">
          <p>
            This RSA implementation follows industry security standards and best practices.
            Always keep your private keys secure and use appropriate key lengths for your security requirements.
          </p>
        </div>
      </div>
      
      <Toaster />
    </div>
  );
};

export default Index;
