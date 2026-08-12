using HRMS.Enum;
using HRMS.Models;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace HRMS.Filters
{
    public class HRPermissionFilter : Attribute, IAuthorizationFilter
    {
        private readonly IJwtService _jwtService;

        public HRPermissionFilter(IJwtService jwtService, ILoggerService loggerService)
        {
            _jwtService = jwtService;
        }

        public void OnAuthorization(AuthorizationFilterContext context)
        {
            if (context.HttpContext.Items.TryGetValue("JwtData", out var jwtData) && jwtData is JwtData userData)
            {
                if (userData.Permission < 1)
                {
                    context.Result = new JsonResult(new ApiBaseResponse<string>(ApiErrorEnum.InvalidPermission, "Permission Denied"));
                    return;
                }
            }
            else
            {
                context.Result = new JsonResult(new ApiBaseResponse<string>(ApiErrorEnum.InvalidPermission, "Permission Denied"));
                return;
            }
        }
    }
}
