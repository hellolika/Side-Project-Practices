using HRMS.Services.Interfaces;
using System.Text;
using XSystem.Security.Cryptography;

namespace HRMS.Services
{
    public class Sha512Service : ISha512Service
    {
        private const string Key = "719D2EB5-CDD1-4370-B2D0-B8C4F5ECBCB9";

        public string Encrypt(string input)
        {
            var bytes = Encoding.Unicode.GetBytes(string.Concat(Key, input));
            return Convert.ToBase64String(new SHA512Managed().ComputeHash(bytes));
        }
    }
}
