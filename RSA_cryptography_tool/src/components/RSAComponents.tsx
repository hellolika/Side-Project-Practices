import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { RSA, RSAKeyPair, RSAPublicKey, RSAPrivateKey, RSASignature } from '@/lib/rsa';
import { Key, Lock, Unlock, FileKey, Download, Upload, Shield, CheckCircle, AlertCircle } from 'lucide-react';

interface KeyGenerationComponentProps {
  onKeyGenerated: (keyPair: RSAKeyPair) => void;
}

const KeyGenerationComponent: React.FC<KeyGenerationComponentProps> = ({ onKeyGenerated }) => {
  const [keySize, setKeySize] = useState<number>(2048);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  const generateKeys = useCallback(async () => {
    setIsGenerating(true);
    setProgress(0);

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 100);

      // Generate keys in a setTimeout to allow UI updates
      setTimeout(() => {
        try {
          const keyPair = RSA.generateKeyPair(keySize);
          clearInterval(progressInterval);
          setProgress(100);
          
          onKeyGenerated(keyPair);
          toast({
            title: "Keys Generated Successfully",
            description: `${keySize}-bit RSA key pair generated with secure random primes.`,
          });
        } catch (error) {
          clearInterval(progressInterval);
          toast({
            title: "Key Generation Failed",
            description: error instanceof Error ? error.message : "Unknown error occurred",
            variant: "destructive",
          });
        } finally {
          setIsGenerating(false);
          setProgress(0);
        }
      }, 1000);
    } catch (error) {
      setIsGenerating(false);
      setProgress(0);
      toast({
        title: "Key Generation Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
    }
  }, [keySize, onKeyGenerated, toast]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Key className="h-5 w-5" />
          RSA Key Generation
        </CardTitle>
        <CardDescription>
          Generate a secure RSA key pair with cryptographically strong prime numbers
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="keySize">Key Size (bits)</Label>
          <Select value={keySize.toString()} onValueChange={(value) => setKeySize(parseInt(value))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="2048">2048 bits (Recommended)</SelectItem>
              <SelectItem value="3072">3072 bits (High Security)</SelectItem>
              <SelectItem value="4096">4096 bits (Maximum Security)</SelectItem>
            </SelectContent>
          </Select>
          <div className="text-sm text-muted-foreground">
            {keySize === 2048 && "Standard security level, suitable for most applications"}
            {keySize === 3072 && "Enhanced security, recommended for sensitive data"}
            {keySize === 4096 && "Maximum security, may be slower for operations"}
          </div>
        </div>

        {isGenerating && (
          <div className="space-y-2">
            <Label>Generation Progress</Label>
            <Progress value={progress} className="w-full" />
            <div className="text-sm text-muted-foreground">
              Generating secure prime numbers and computing key parameters...
            </div>
          </div>
        )}

        <Alert>
          <Shield className="h-4 w-4" />
          <AlertDescription>
            Keys are generated using cryptographically secure random number generation (CSPRNG) 
            and Miller-Rabin primality testing for maximum security.
          </AlertDescription>
        </Alert>

        <Button 
          onClick={generateKeys} 
          disabled={isGenerating}
          className="w-full bg-[rgb(0,179,141)] hover:bg-[rgb(0,179,141)]/90 text-white"
        >
          {isGenerating ? "Generating Keys..." : "Generate RSA Key Pair"}
        </Button>
      </CardContent>
    </Card>
  );
};

interface EncryptionComponentProps {
  publicKey: RSAPublicKey | null;
}

const EncryptionComponent: React.FC<EncryptionComponentProps> = ({ publicKey }) => {
  const [plaintext, setPlaintext] = useState('');
  const [ciphertext, setCiphertext] = useState('');
  const [useOAEP, setUseOAEP] = useState(true);
  const [isEncrypting, setIsEncrypting] = useState(false);
  const { toast } = useToast();

  const encrypt = useCallback(async () => {
    if (!publicKey) {
      toast({
        title: "No Public Key",
        description: "Please generate or import a public key first.",
        variant: "destructive",
      });
      return;
    }

    if (!plaintext.trim()) {
      toast({
        title: "No Message",
        description: "Please enter a message to encrypt.",
        variant: "destructive",
      });
      return;
    }

    setIsEncrypting(true);
    try {
      const encrypted = RSA.encrypt(plaintext, publicKey, useOAEP);
      setCiphertext(encrypted);
      toast({
        title: "Encryption Successful",
        description: `Message encrypted using ${useOAEP ? 'OAEP' : 'PKCS#1 v1.5'} padding.`,
      });
    } catch (error) {
      toast({
        title: "Encryption Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
    } finally {
      setIsEncrypting(false);
    }
  }, [plaintext, publicKey, useOAEP, toast]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Lock className="h-5 w-5" />
          Public Key Encryption
        </CardTitle>
        <CardDescription>
          Encrypt messages using the public key with secure padding schemes
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="plaintext">Message to Encrypt</Label>
          <Textarea
            id="plaintext"
            placeholder="Enter your message here..."
            value={plaintext}
            onChange={(e) => setPlaintext(e.target.value)}
            rows={4}
          />
        </div>

        <div className="flex items-center space-x-2">
          <Switch
            id="padding"
            checked={useOAEP}
            onCheckedChange={setUseOAEP}
          />
          <Label htmlFor="padding">
            Use OAEP Padding (Recommended)
          </Label>
        </div>
        <div className="text-sm text-muted-foreground">
          {useOAEP 
            ? "OAEP provides better security against chosen-ciphertext attacks"
            : "PKCS#1 v1.5 padding for compatibility with older systems"
          }
        </div>

        <Button 
          onClick={encrypt} 
          disabled={!publicKey || isEncrypting}
          className="w-full"
        >
          {isEncrypting ? "Encrypting..." : "Encrypt Message"}
        </Button>

        {ciphertext && (
          <div className="space-y-2">
            <Label>Encrypted Message (Hexadecimal)</Label>
            <Textarea
              value={ciphertext}
              readOnly
              rows={6}
              className="font-mono text-sm"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigator.clipboard.writeText(ciphertext)}
            >
              Copy to Clipboard
            </Button>
          </div>
        )}

        {!publicKey && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Generate or import a public key to enable encryption.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

interface DecryptionComponentProps {
  privateKey: RSAPrivateKey | null;
}

const DecryptionComponent: React.FC<DecryptionComponentProps> = ({ privateKey }) => {
  const [ciphertext, setCiphertext] = useState('');
  const [plaintext, setPlaintext] = useState('');
  const [useOAEP, setUseOAEP] = useState(true);
  const [isDecrypting, setIsDecrypting] = useState(false);
  const { toast } = useToast();

  const decrypt = useCallback(async () => {
    if (!privateKey) {
      toast({
        title: "No Private Key",
        description: "Please generate or import a private key first.",
        variant: "destructive",
      });
      return;
    }

    if (!ciphertext.trim()) {
      toast({
        title: "No Ciphertext",
        description: "Please enter a ciphertext to decrypt.",
        variant: "destructive",
      });
      return;
    }

    setIsDecrypting(true);
    try {
      const decrypted = RSA.decrypt(ciphertext, privateKey, useOAEP);
      setPlaintext(decrypted);
      toast({
        title: "Decryption Successful",
        description: `Message decrypted using ${useOAEP ? 'OAEP' : 'PKCS#1 v1.5'} padding.`,
      });
    } catch (error) {
      toast({
        title: "Decryption Failed",
        description: error instanceof Error ? error.message : "Invalid ciphertext or wrong key",
        variant: "destructive",
      });
    } finally {
      setIsDecrypting(false);
    }
  }, [ciphertext, privateKey, useOAEP, toast]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Unlock className="h-5 w-5" />
          Private Key Decryption
        </CardTitle>
        <CardDescription>
          Decrypt ciphertext using the private key
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="ciphertext">Ciphertext to Decrypt (Hexadecimal)</Label>
          <Textarea
            id="ciphertext"
            placeholder="Enter the encrypted message in hexadecimal format..."
            value={ciphertext}
            onChange={(e) => setCiphertext(e.target.value)}
            rows={6}
            className="font-mono text-sm"
          />
        </div>

        <div className="flex items-center space-x-2">
          <Switch
            id="padding-decrypt"
            checked={useOAEP}
            onCheckedChange={setUseOAEP}
          />
          <Label htmlFor="padding-decrypt">
            Use OAEP Padding
          </Label>
        </div>

        <Button 
          onClick={decrypt} 
          disabled={!privateKey || isDecrypting}
          className="w-full"
        >
          {isDecrypting ? "Decrypting..." : "Decrypt Message"}
        </Button>

        {plaintext && (
          <div className="space-y-2">
            <Label>Decrypted Message</Label>
            <Textarea
              value={plaintext}
              readOnly
              rows={4}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigator.clipboard.writeText(plaintext)}
            >
              Copy to Clipboard
            </Button>
          </div>
        )}

        {!privateKey && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Generate or import a private key to enable decryption.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

interface SigningComponentProps {
  privateKey: RSAPrivateKey | null;
}

const SigningComponent: React.FC<SigningComponentProps> = ({ privateKey }) => {
  const [message, setMessage] = useState('');
  const [signature, setSignature] = useState<RSASignature | null>(null);
  const [isSigning, setIsSigning] = useState(false);
  const { toast } = useToast();

  const signMessage = useCallback(async () => {
    if (!privateKey) {
      toast({
        title: "No Private Key",
        description: "Please generate or import a private key first.",
        variant: "destructive",
      });
      return;
    }

    if (!message.trim()) {
      toast({
        title: "No Message",
        description: "Please enter a message to sign.",
        variant: "destructive",
      });
      return;
    }

    setIsSigning(true);
    try {
      const sig = RSA.sign(message, privateKey);
      setSignature(sig);
      toast({
        title: "Signing Successful",
        description: "Message signed with your private key.",
      });
    } catch (error) {
      toast({
        title: "Signing Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
    } finally {
      setIsSigning(false);
    }
  }, [message, privateKey, toast]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileKey className="h-5 w-5" />
          Digital Signing
        </CardTitle>
        <CardDescription>
          Sign messages with your private key to ensure authenticity and integrity
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="message-sign">Message to Sign</Label>
          <Textarea
            id="message-sign"
            placeholder="Enter your message here..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={4}
          />
        </div>

        <Button 
          onClick={signMessage} 
          disabled={!privateKey || isSigning}
          className="w-full"
        >
          {isSigning ? "Signing..." : "Sign Message"}
        </Button>

        {signature && (
          <div className="space-y-2">
            <Label>Digital Signature (Hexadecimal)</Label>
            <Textarea
              value={signature.signature.toString(16)}
              readOnly
              rows={6}
              className="font-mono text-sm"
            />
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigator.clipboard.writeText(signature.signature.toString(16))}
              >
                Copy Signature
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigator.clipboard.writeText(JSON.stringify({
                  message: signature.message,
                  signature: signature.signature.toString(16)
                }))}
              >
                Copy Full Package
              </Button>
            </div>
          </div>
        )}

        {!privateKey && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Generate or import a private key to enable message signing.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

interface VerificationComponentProps {
  publicKey: RSAPublicKey | null;
}

const VerificationComponent: React.FC<VerificationComponentProps> = ({ publicKey }) => {
  const [message, setMessage] = useState('');
  const [signatureHex, setSignatureHex] = useState('');
  const [verificationResult, setVerificationResult] = useState<boolean | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const { toast } = useToast();

  const verifySignature = useCallback(async () => {
    if (!publicKey) {
      toast({
        title: "No Public Key",
        description: "Please generate or import a public key first.",
        variant: "destructive",
      });
      return;
    }

    if (!message.trim() || !signatureHex.trim()) {
      toast({
        title: "Missing Information",
        description: "Please enter both the message and signature.",
        variant: "destructive",
      });
      return;
    }

    setIsVerifying(true);
    try {
      const signature: RSASignature = {
        message,
        signature: BigInt('0x' + signatureHex)
      };
      
      const isValid = RSA.verify(signature, publicKey);
      setVerificationResult(isValid);
      
      toast({
        title: isValid ? "Signature Valid" : "Signature Invalid",
        description: isValid 
          ? "The signature is authentic and the message has not been tampered with."
          : "The signature is invalid or the message has been modified.",
        variant: isValid ? "default" : "destructive",
      });
    } catch (error) {
      setVerificationResult(false);
      toast({
        title: "Verification Failed",
        description: error instanceof Error ? error.message : "Invalid signature format",
        variant: "destructive",
      });
    } finally {
      setIsVerifying(false);
    }
  }, [message, signatureHex, publicKey, toast]);

  const loadFromPackage = useCallback(() => {
    try {
      const packageData = JSON.parse(signatureHex);
      if (packageData.message && packageData.signature) {
        setMessage(packageData.message);
        setSignatureHex(packageData.signature);
        toast({
          title: "Package Loaded",
          description: "Message and signature loaded from package.",
        });
      }
    } catch {
      toast({
        title: "Invalid Package",
        description: "Could not parse the signature package.",
        variant: "destructive",
      });
    }
  }, [signatureHex, toast]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CheckCircle className="h-5 w-5" />
          Signature Verification
        </CardTitle>
        <CardDescription>
          Verify digital signatures using the public key
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="message-verify">Original Message</Label>
          <Textarea
            id="message-verify"
            placeholder="Enter the original message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={4}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="signature-verify">Signature (Hexadecimal or JSON Package)</Label>
          <Textarea
            id="signature-verify"
            placeholder="Enter the signature in hexadecimal format or paste a JSON package..."
            value={signatureHex}
            onChange={(e) => setSignatureHex(e.target.value)}
            rows={6}
            className="font-mono text-sm"
          />
          <Button
            variant="outline"
            size="sm"
            onClick={loadFromPackage}
          >
            Load from JSON Package
          </Button>
        </div>

        <Button 
          onClick={verifySignature} 
          disabled={!publicKey || isVerifying}
          className="w-full"
        >
          {isVerifying ? "Verifying..." : "Verify Signature"}
        </Button>

        {verificationResult !== null && (
          <Alert className={verificationResult ? "border-green-500" : "border-red-500"}>
            {verificationResult ? (
              <CheckCircle className="h-4 w-4 text-green-500" />
            ) : (
              <AlertCircle className="h-4 w-4 text-red-500" />
            )}
            <AlertDescription>
              {verificationResult 
                ? "✅ Signature is valid - Message is authentic and unmodified"
                : "❌ Signature is invalid - Message may have been tampered with or signature is incorrect"
              }
            </AlertDescription>
          </Alert>
        )}

        {!publicKey && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Generate or import a public key to enable signature verification.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

interface KeyManagementComponentProps {
  keyPair: RSAKeyPair | null;
  onKeyImported: (keyPair: Partial<RSAKeyPair>) => void;
}

const KeyManagementComponent: React.FC<KeyManagementComponentProps> = ({ keyPair, onKeyImported }) => {
  const [importData, setImportData] = useState('');
  const [importType, setImportType] = useState<'public' | 'private'>('public');
  const { toast } = useToast();

  const exportKey = useCallback((type: 'public' | 'private') => {
    if (!keyPair) {
      toast({
        title: "No Keys Available",
        description: "Please generate keys first.",
        variant: "destructive",
      });
      return;
    }

    try {
      let exportData: string;
      let filename: string;

      if (type === 'public' && keyPair.publicKey) {
        exportData = RSA.exportPublicKey(keyPair.publicKey);
        filename = 'rsa_public_key.json';
      } else if (type === 'private' && keyPair.privateKey) {
        exportData = RSA.exportPrivateKey(keyPair.privateKey);
        filename = 'rsa_private_key.json';
      } else {
        throw new Error('Key not available');
      }

      // Create download
      const blob = new Blob([exportData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast({
        title: "Key Exported",
        description: `${type === 'public' ? 'Public' : 'Private'} key exported successfully.`,
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
    }
  }, [keyPair, toast]);

  const importKey = useCallback(() => {
    if (!importData.trim()) {
      toast({
        title: "No Data",
        description: "Please enter key data to import.",
        variant: "destructive",
      });
      return;
    }

    try {
      if (importType === 'public') {
        const publicKey = RSA.importPublicKey(importData);
        onKeyImported({ publicKey });
        toast({
          title: "Public Key Imported",
          description: "Public key imported successfully.",
        });
      } else {
        const privateKey = RSA.importPrivateKey(importData);
        // Derive public key from private key
        const publicKey: RSAPublicKey = {
          n: privateKey.n,
          e: 65537n, // Standard public exponent
          keySize: privateKey.keySize
        };
        onKeyImported({ publicKey, privateKey });
        toast({
          title: "Private Key Imported",
          description: "Private key and derived public key imported successfully.",
        });
      }
      setImportData('');
    } catch (error) {
      toast({
        title: "Import Failed",
        description: error instanceof Error ? error.message : "Invalid key format",
        variant: "destructive",
      });
    }
  }, [importData, importType, onKeyImported, toast]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileKey className="h-5 w-5" />
          Key Management
        </CardTitle>
        <CardDescription>
          Export and import RSA keys securely
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Keys Status */}
        <div className="space-y-2">
          <Label>Current Keys Status</Label>
          <div className="flex gap-2">
            <Badge variant={keyPair?.publicKey ? "default" : "secondary"}>
              {keyPair?.publicKey ? "✅ Public Key" : "❌ No Public Key"}
            </Badge>
            <Badge variant={keyPair?.privateKey ? "default" : "secondary"}>
              {keyPair?.privateKey ? "✅ Private Key" : "❌ No Private Key"}
            </Badge>
          </div>
          {keyPair?.publicKey && (
            <div className="text-sm text-muted-foreground">
              Key Size: {keyPair.publicKey.keySize} bits
            </div>
          )}
        </div>

        {/* Export Keys */}
        <div className="space-y-2">
          <Label>Export Keys</Label>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => exportKey('public')}
              disabled={!keyPair?.publicKey}
              className="flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Export Public Key
            </Button>
            <Button
              variant="outline"
              onClick={() => exportKey('private')}
              disabled={!keyPair?.privateKey}
              className="flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Export Private Key
            </Button>
          </div>
        </div>

        {/* Import Keys */}
        <div className="space-y-4">
          <Label>Import Keys</Label>
          
          <div className="space-y-2">
            <Label htmlFor="import-type">Key Type</Label>
            <Select value={importType} onValueChange={(value: 'public' | 'private') => setImportType(value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="public">Public Key</SelectItem>
                <SelectItem value="private">Private Key</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="import-data">Key Data (JSON)</Label>
            <Textarea
              id="import-data"
              placeholder="Paste the exported key data here..."
              value={importData}
              onChange={(e) => setImportData(e.target.value)}
              rows={8}
              className="font-mono text-sm"
            />
          </div>

          <Button
            onClick={importKey}
            className="w-full flex items-center gap-2"
          >
            <Upload className="h-4 w-4" />
            Import {importType === 'public' ? 'Public' : 'Private'} Key
          </Button>
        </div>

        <Alert>
          <Shield className="h-4 w-4" />
          <AlertDescription>
            <strong>Security Warning:</strong> Keep your private keys secure and never share them. 
            Only share public keys for encryption and signature verification.
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
};

export {
  KeyGenerationComponent,
  EncryptionComponent,
  DecryptionComponent,
  SigningComponent,
  VerificationComponent,
  KeyManagementComponent
};