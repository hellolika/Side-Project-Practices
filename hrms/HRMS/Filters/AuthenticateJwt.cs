using HRMS.Enum;
using HRMS.Models;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace HRMS.Filters
{
    public class AuthenticateJwt : Attribute, IAuthorizationFilter
    {
        private readonly ILoggerService _loggerService;
        private readonly IAuthService _authService;

        public AuthenticateJwt(ILoggerService loggerService, IAuthService authService)
        {
            _loggerService = loggerService;
            _authService = authService;
        }

        public void OnAuthorization(AuthorizationFilterContext context)
        {
            string token = context.HttpContext.Request.Headers["Authorization"];

            if (string.IsNullOrEmpty(token))
            {
                SetContextResult(ApiErrorEnum.AuthorizationFailed, "Can't found token in Request header", context);
                return;
            }

            token = token.Replace("Bearer ", "");

            if (!_authService.TryDecryptToken(token, out JwtData jwtDataResult))
            {
                SetContextResult(ApiErrorEnum.InvalidToken, "Token decrypt failed", context);
                return;
            }

            if (!_authService.IsMemberExist(jwtDataResult.Id))
            {
                SetContextResult(ApiErrorEnum.MemberNotFound, "Can't find members", context);
                return;
            }

            context.HttpContext.Items.Add("JwtData", jwtDataResult);
        }

        private void SetContextResult(ApiErrorEnum code, string message, AuthorizationFilterContext context)
        {
            _loggerService.Error($"[Authenticate Filter] Error: {code}, Message: {message}");
            int statusCode = code == ApiErrorEnum.MemberNotFound ? 200 : 403;
            context.Result = new JsonResult(new ApiBaseResponse<string>(code, "Unauthorized")) { StatusCode = statusCode };
        }
    }
}