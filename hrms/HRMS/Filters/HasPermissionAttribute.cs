using HRMS.Enum;
using HRMS.Models;
using HRMS.Models.Responses;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace HRMS.Filters;

public class HasPermissionAttribute : Attribute, IAuthorizationFilter
{
    private readonly PermissionEnum _requiredPermission;
    private readonly PermissionEnum _optionalPermission;

    public HasPermissionAttribute(PermissionEnum claims,PermissionEnum optionalClaim = PermissionEnum.Unknown)
    {
        _requiredPermission = claims;
        _optionalPermission = optionalClaim;
    }
    public void OnAuthorization(AuthorizationFilterContext context)
    {
        if (context.HttpContext.Items.TryGetValue("JwtData", out var jwtData) && jwtData is JwtData userData)
        {

            if (!userData.Permissions.Contains(Convert.ToInt32(_requiredPermission).ToString()) && !userData.Permissions.Contains(Convert.ToInt32(_optionalPermission).ToString()))
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