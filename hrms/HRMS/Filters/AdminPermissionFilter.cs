using HRMS.Enum;
using HRMS.Models;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace HRMS.Filters
{
    public class AdminPermissionFilter : Attribute, IAuthorizationFilter
    {
        private readonly IJwtService _jwtService;

        public AdminPermissionFilter(IJwtService jwtService, ILoggerService loggerService)
        {
            _jwtService = jwtService;
        }

        public void OnAuthorization(AuthorizationFilterContext context)
        {
            if (context.HttpContext.Items.TryGetValue("JwtData", out var jwtData) && jwtData is JwtData userData)
            {
                if (!userData.IsAdmin())
                {
                    context.Result = new JsonResult(new ApiBaseResponse<string>(ApiErrorEnum.InvalidPermission, "Permission Denied"));
                }
            }
            else
            {
                context.Result = new JsonResult(new ApiBaseResponse<string>(ApiErrorEnum.InvalidPermission, "Permission Denied"));
            }
        }
    }
}
