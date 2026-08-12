using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Models;
using HRMS.Services.Interfaces;

namespace HRMS.Services
{
    public class MemberDataService : IMemberDataService
    {
        private readonly IHttpContextAccessor _httpContextAccessor;

        public MemberDataService(IHttpContextAccessor httpContextAccessor)
        {
            _httpContextAccessor = httpContextAccessor;
        }

        public int GetCurrentMemberId()
        {
            return GetCurrentMemberData().Id;
        }

        private JwtData GetCurrentMemberData()
        {
            if (_httpContextAccessor.HttpContext.Items.TryGetValue("JwtData", out var jwtData) && jwtData is JwtData userData)
            {
                return userData;
            }
            throw new ApiException(ApiErrorEnum.AuthorizationFailed, "Cannot get JwtData from HttpContext");
        }
    }
}
