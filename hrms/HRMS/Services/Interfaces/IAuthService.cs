using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;

namespace HRMS.Services.Interfaces
{
    public interface IAuthService
    {
        bool IsMemberExist(int id);

        ApiBaseResponse<LoginResponse> Login(LoginRequest request);

        bool TryDecryptToken(string token, out JwtData jwtDataResult);
    }
}