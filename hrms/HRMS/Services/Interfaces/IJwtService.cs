using HRMS.Models;

namespace HRMS.Services.Interfaces
{
    public interface IJwtService
    {
        string GenerateToken(JwtData data);
        JwtData DecryptToken(string token, bool validateLifetime = false);
    }
}
