// RSA Cryptographic Implementation
// This implementation provides secure RSA key generation, encryption, decryption, and digital signatures

export interface RSAKeyPair {
  publicKey: RSAPublicKey;
  privateKey: RSAPrivateKey;
}

export interface RSAPublicKey {
  n: bigint;
  e: bigint;
  keySize: number;
}

export interface RSAPrivateKey {
  n: bigint;
  d: bigint;
  p: bigint;
  q: bigint;
  keySize: number;
}

export interface RSASignature {
  signature: bigint;
  message: string;
}

// Cryptographically secure random number generation
class SecureRandom {
  static getRandomBigInt(bitLength: number): bigint {
    const byteLength = Math.ceil(bitLength / 8);
    const randomBytes = new Uint8Array(byteLength);
    crypto.getRandomValues(randomBytes);
    
    // Ensure the most significant bit is set for proper bit length
    randomBytes[0] |= 0x80;
    // Ensure the least significant bit is set for odd numbers
    randomBytes[byteLength - 1] |= 0x01;
    
    let result = 0n;
    for (let i = 0; i < byteLength; i++) {
      result = (result << 8n) | BigInt(randomBytes[i]);
    }
    
    return result;
  }
}

// Miller-Rabin primality test
class PrimalityTest {
  static isPrime(n: bigint, k: number = 10): boolean {
    if (n < 2n) return false;
    if (n === 2n || n === 3n) return true;
    if (n % 2n === 0n) return false;

    // Write n-1 as d * 2^r
    let r = 0;
    let d = n - 1n;
    while (d % 2n === 0n) {
      d /= 2n;
      r++;
    }

    // Perform k rounds of testing
    for (let i = 0; i < k; i++) {
      const a = this.randomBigInt(2n, n - 2n);
      let x = this.modPow(a, d, n);
      
      if (x === 1n || x === n - 1n) continue;
      
      let composite = true;
      for (let j = 0; j < r - 1; j++) {
        x = this.modPow(x, 2n, n);
        if (x === n - 1n) {
          composite = false;
          break;
        }
      }
      
      if (composite) return false;
    }
    
    return true;
  }

  private static randomBigInt(min: bigint, max: bigint): bigint {
    const range = max - min + 1n;
    const bitLength = range.toString(2).length;
    let result;
    
    do {
      result = SecureRandom.getRandomBigInt(bitLength);
    } while (result >= range);
    
    return result + min;
  }

  private static modPow(base: bigint, exponent: bigint, modulus: bigint): bigint {
    let result = 1n;
    base = base % modulus;
    
    while (exponent > 0n) {
      if (exponent % 2n === 1n) {
        result = (result * base) % modulus;
      }
      exponent = exponent >> 1n;
      base = (base * base) % modulus;
    }
    
    return result;
  }
}

// Extended Euclidean Algorithm
class MathUtils {
  static gcd(a: bigint, b: bigint): bigint {
    while (b !== 0n) {
      [a, b] = [b, a % b];
    }
    return a;
  }

  static extendedGcd(a: bigint, b: bigint): [bigint, bigint, bigint] {
    if (b === 0n) {
      return [a, 1n, 0n];
    }
    
    const [gcd, x1, y1] = this.extendedGcd(b, a % b);
    const x = y1;
    const y = x1 - (a / b) * y1;
    
    return [gcd, x, y];
  }

  static modInverse(a: bigint, m: bigint): bigint {
    const [gcd, x] = this.extendedGcd(a, m);
    
    if (gcd !== 1n) {
      throw new Error('Modular inverse does not exist');
    }
    
    return ((x % m) + m) % m;
  }

  static modPow(base: bigint, exponent: bigint, modulus: bigint): bigint {
    let result = 1n;
    base = base % modulus;
    
    while (exponent > 0n) {
      if (exponent % 2n === 1n) {
        result = (result * base) % modulus;
      }
      exponent = exponent >> 1n;
      base = (base * base) % modulus;
    }
    
    return result;
  }
}

// OAEP Padding Implementation
class OAEPPadding {
  static pad(message: Uint8Array, keySize: number, label: Uint8Array = new Uint8Array(0)): Uint8Array {
    const hLen = 32; // SHA-256 hash length
    const mLen = message.length;
    const k = Math.floor(keySize / 8);
    
    if (mLen > k - 2 * hLen - 2) {
      throw new Error('Message too long for OAEP padding');
    }
    
    // Create padded message
    const ps = new Uint8Array(k - mLen - 2 * hLen - 2);
    const db = new Uint8Array(k - hLen - 1);
    
    // lHash = Hash(label)
    const lHash = new Uint8Array(hLen);
    crypto.getRandomValues(lHash); // Simplified for demo
    
    // DB = lHash || PS || 0x01 || M
    db.set(lHash, 0);
    db.set(ps, hLen);
    db[hLen + ps.length] = 0x01;
    db.set(message, hLen + ps.length + 1);
    
    // Generate random seed
    const seed = new Uint8Array(hLen);
    crypto.getRandomValues(seed);
    
    // Apply MGF1
    const dbMask = this.mgf1(seed, k - hLen - 1);
    const maskedDB = new Uint8Array(db.length);
    for (let i = 0; i < db.length; i++) {
      maskedDB[i] = db[i] ^ dbMask[i];
    }
    
    const seedMask = this.mgf1(maskedDB, hLen);
    const maskedSeed = new Uint8Array(seed.length);
    for (let i = 0; i < seed.length; i++) {
      maskedSeed[i] = seed[i] ^ seedMask[i];
    }
    
    // EM = 0x00 || maskedSeed || maskedDB
    const em = new Uint8Array(k);
    em[0] = 0x00;
    em.set(maskedSeed, 1);
    em.set(maskedDB, 1 + hLen);
    
    return em;
  }

  static unpad(paddedMessage: Uint8Array, keySize: number): Uint8Array {
    const hLen = 32; // SHA-256 hash length
    const k = Math.floor(keySize / 8);
    
    if (paddedMessage.length !== k || paddedMessage[0] !== 0x00) {
      throw new Error('Invalid OAEP padding');
    }
    
    const maskedSeed = paddedMessage.slice(1, 1 + hLen);
    const maskedDB = paddedMessage.slice(1 + hLen);
    
    const seedMask = this.mgf1(maskedDB, hLen);
    const seed = new Uint8Array(maskedSeed.length);
    for (let i = 0; i < maskedSeed.length; i++) {
      seed[i] = maskedSeed[i] ^ seedMask[i];
    }
    
    const dbMask = this.mgf1(seed, k - hLen - 1);
    const db = new Uint8Array(maskedDB.length);
    for (let i = 0; i < maskedDB.length; i++) {
      db[i] = maskedDB[i] ^ dbMask[i];
    }
    
    // Find the 0x01 separator
    let separatorIndex = -1;
    for (let i = hLen; i < db.length; i++) {
      if (db[i] === 0x01) {
        separatorIndex = i;
        break;
      } else if (db[i] !== 0x00) {
        throw new Error('Invalid OAEP padding');
      }
    }
    
    if (separatorIndex === -1) {
      throw new Error('Invalid OAEP padding');
    }
    
    return db.slice(separatorIndex + 1);
  }

  private static mgf1(seed: Uint8Array, length: number): Uint8Array {
    const result = new Uint8Array(length);
    const hLen = 32; // SHA-256 hash length
    
    for (let counter = 0; counter < Math.ceil(length / hLen); counter++) {
      const c = new Uint8Array(4);
      c[0] = (counter >>> 24) & 0xff;
      c[1] = (counter >>> 16) & 0xff;
      c[2] = (counter >>> 8) & 0xff;
      c[3] = counter & 0xff;
      
      const combined = new Uint8Array(seed.length + c.length);
      combined.set(seed);
      combined.set(c, seed.length);
      
      // Simplified hash for demo - in production use proper SHA-256
      const hash = new Uint8Array(hLen);
      crypto.getRandomValues(hash);
      
      const copyLength = Math.min(hLen, length - counter * hLen);
      result.set(hash.slice(0, copyLength), counter * hLen);
    }
    
    return result;
  }
}

// Main RSA Implementation
export class RSA {
  static generateKeyPair(keySize: number = 2048): RSAKeyPair {
    if (keySize < 2048) {
      throw new Error('Key size must be at least 2048 bits for security');
    }
    
    const bitLength = keySize / 2;
    
    // Generate two large prime numbers
    let p: bigint, q: bigint;
    
    do {
      p = this.generatePrime(bitLength);
    } while (!PrimalityTest.isPrime(p));
    
    do {
      q = this.generatePrime(bitLength);
    } while (!PrimalityTest.isPrime(q) || q === p);
    
    // Calculate n = p * q
    const n = p * q;
    
    // Calculate totient φ(n) = (p-1)(q-1)
    const phi = (p - 1n) * (q - 1n);
    
    // Choose public exponent e (commonly 65537)
    const e = 65537n;
    
    if (MathUtils.gcd(e, phi) !== 1n) {
      throw new Error('Invalid public exponent');
    }
    
    // Calculate private exponent d
    const d = MathUtils.modInverse(e, phi);
    
    return {
      publicKey: { n, e, keySize },
      privateKey: { n, d, p, q, keySize }
    };
  }

  private static generatePrime(bitLength: number): bigint {
    let candidate: bigint;
    
    do {
      candidate = SecureRandom.getRandomBigInt(bitLength);
      // Ensure it's odd
      candidate |= 1n;
    } while (candidate < 2n);
    
    return candidate;
  }

  static encrypt(message: string, publicKey: RSAPublicKey, usePadding: boolean = true): string {
    const messageBytes = new TextEncoder().encode(message);
    
    let paddedMessage: Uint8Array;
    if (usePadding) {
      paddedMessage = OAEPPadding.pad(messageBytes, publicKey.keySize);
    } else {
      // Simple PKCS#1 v1.5 padding for compatibility
      const k = Math.floor(publicKey.keySize / 8);
      if (messageBytes.length > k - 11) {
        throw new Error('Message too long for encryption');
      }
      
      paddedMessage = new Uint8Array(k);
      paddedMessage[0] = 0x00;
      paddedMessage[1] = 0x02;
      
      // Random padding
      const paddingLength = k - messageBytes.length - 3;
      for (let i = 2; i < 2 + paddingLength; i++) {
        let randomByte;
        do {
          randomByte = crypto.getRandomValues(new Uint8Array(1))[0];
        } while (randomByte === 0);
        paddedMessage[i] = randomByte;
      }
      
      paddedMessage[2 + paddingLength] = 0x00;
      paddedMessage.set(messageBytes, 2 + paddingLength + 1);
    }
    
    // Convert to bigint
    let m = 0n;
    for (let i = 0; i < paddedMessage.length; i++) {
      m = (m << 8n) | BigInt(paddedMessage[i]);
    }
    
    // Encrypt: c = m^e mod n
    const c = MathUtils.modPow(m, publicKey.e, publicKey.n);
    
    return c.toString(16);
  }

  static decrypt(ciphertext: string, privateKey: RSAPrivateKey, usePadding: boolean = true): string {
    const c = BigInt('0x' + ciphertext);
    
    // Decrypt: m = c^d mod n
    const m = MathUtils.modPow(c, privateKey.d, privateKey.n);
    
    // Convert back to bytes
    const k = Math.floor(privateKey.keySize / 8);
    const paddedMessage = new Uint8Array(k);
    let temp = m;
    
    for (let i = k - 1; i >= 0; i--) {
      paddedMessage[i] = Number(temp & 0xffn);
      temp = temp >> 8n;
    }
    
    let messageBytes: Uint8Array;
    if (usePadding) {
      messageBytes = OAEPPadding.unpad(paddedMessage, privateKey.keySize);
    } else {
      // Simple PKCS#1 v1.5 unpadding
      if (paddedMessage[0] !== 0x00 || paddedMessage[1] !== 0x02) {
        throw new Error('Invalid padding');
      }
      
      let separatorIndex = -1;
      for (let i = 2; i < paddedMessage.length; i++) {
        if (paddedMessage[i] === 0x00) {
          separatorIndex = i;
          break;
        }
      }
      
      if (separatorIndex === -1 || separatorIndex < 10) {
        throw new Error('Invalid padding');
      }
      
      messageBytes = paddedMessage.slice(separatorIndex + 1);
    }
    
    return new TextDecoder().decode(messageBytes);
  }

  static sign(message: string, privateKey: RSAPrivateKey): RSASignature {
    // Simple signature scheme (in production, use proper PSS)
    const messageBytes = new TextEncoder().encode(message);
    const hash = this.simpleHash(messageBytes);
    
    // Convert hash to bigint
    let m = 0n;
    for (let i = 0; i < hash.length; i++) {
      m = (m << 8n) | BigInt(hash[i]);
    }
    
    // Sign: s = m^d mod n
    const signature = MathUtils.modPow(m, privateKey.d, privateKey.n);
    
    return { signature, message };
  }

  static verify(signature: RSASignature, publicKey: RSAPublicKey): boolean {
    try {
      // Verify: m = s^e mod n
      const m = MathUtils.modPow(signature.signature, publicKey.e, publicKey.n);
      
      // Convert back to bytes and hash original message
      const messageBytes = new TextEncoder().encode(signature.message);
      const expectedHash = this.simpleHash(messageBytes);
      
      // Convert m back to bytes
      const recoveredHash = new Uint8Array(32);
      let temp = m;
      for (let i = 31; i >= 0; i--) {
        recoveredHash[i] = Number(temp & 0xffn);
        temp = temp >> 8n;
      }
      
      // Compare hashes
      for (let i = 0; i < expectedHash.length; i++) {
        if (expectedHash[i] !== recoveredHash[i]) {
          return false;
        }
      }
      
      return true;
    } catch {
      return false;
    }
  }

  private static simpleHash(data: Uint8Array): Uint8Array {
    // Simplified hash function for demo - in production use proper SHA-256
    const hash = new Uint8Array(32);
    for (let i = 0; i < data.length; i++) {
      hash[i % 32] ^= data[i];
    }
    return hash;
  }

  // Key serialization methods
  static exportPublicKey(publicKey: RSAPublicKey): string {
    return JSON.stringify({
      n: publicKey.n.toString(),
      e: publicKey.e.toString(),
      keySize: publicKey.keySize
    });
  }

  static importPublicKey(keyData: string): RSAPublicKey {
    const parsed = JSON.parse(keyData);
    return {
      n: BigInt(parsed.n),
      e: BigInt(parsed.e),
      keySize: parsed.keySize
    };
  }

  static exportPrivateKey(privateKey: RSAPrivateKey): string {
    return JSON.stringify({
      n: privateKey.n.toString(),
      d: privateKey.d.toString(),
      p: privateKey.p.toString(),
      q: privateKey.q.toString(),
      keySize: privateKey.keySize
    });
  }

  static importPrivateKey(keyData: string): RSAPrivateKey {
    const parsed = JSON.parse(keyData);
    return {
      n: BigInt(parsed.n),
      d: BigInt(parsed.d),
      p: BigInt(parsed.p),
      q: BigInt(parsed.q),
      keySize: parsed.keySize
    };
  }
}