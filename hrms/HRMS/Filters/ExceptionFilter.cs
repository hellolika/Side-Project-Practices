using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace HRMS.Filters
{
    public class ExceptionFilter : ActionFilterAttribute
    {
        private readonly ILoggerService _loggerService;

        public ExceptionFilter(ILoggerService loggerService)
        {
            _loggerService = loggerService;
        }

        public override void OnActionExecuted(ActionExecutedContext context)
        {
            if (context.Exception != null)
            {
                _loggerService.Error(
                    $"[{context.Controller.GetType().Name}Controller] Api occurred exception : {context.Exception.Message}, Stack {context.Exception.StackTrace}");

                var errorCode = context.Exception is ApiException exception
                    ? exception.Error
                    : ApiErrorEnum.InternalError;

                context.Result = new ObjectResult(new ApiBaseResponse<string>(errorCode, context.Exception.Message));
                context.ExceptionHandled = true;
            }
        }
    }
}
