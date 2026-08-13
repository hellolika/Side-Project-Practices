using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc.Filters;
using Newtonsoft.Json;
using System.Text;

namespace HRMS.Filters
{
    public class LogFilter : ActionFilterAttribute
    {
        private readonly ILoggerService _loggerService;

        public LogFilter(ILoggerService loggerService)
        {
            _loggerService = loggerService;
        }

        public override void OnActionExecuting(ActionExecutingContext context)
        {
            var hashCode = RandomString(8);

            try
            {
                _loggerService.Info(
                    $"[#{hashCode}] Request url => {context.HttpContext.Request.Path.ToUriComponent()} => {GetRequestBody(context.ActionArguments)}");
                context.HttpContext.Request.Headers.Add("hashtoken", hashCode);
            }
            catch (Exception)
            {
                // ignored
            }
        }

        public override void OnActionExecuted(ActionExecutedContext context)
        {
            try
            {
                if (context.Exception != null)
                {
                    if (context.Exception.Message.Contains("LoginByToken"))
                    {
                        _loggerService.Error($"message:{context.Exception.Message}");
                    }
                    else
                    {
                        _loggerService.Error(
                            $"Response url: {context.HttpContext.Request.Path.ToUriComponent()}, message:{context.Exception.Message}, callstack: {context.Exception.StackTrace}");
                    }
                }

                var hashCode = context.HttpContext.Request.Headers["hashtoken"];
                var message = context.Result;
                var response = message?.GetType().GetProperty("Value")?.GetValue(message, null);
                _loggerService.Info(
                    $"[#{hashCode}] Response url => {context.HttpContext.Request.Path.ToUriComponent()} => {JsonConvert.SerializeObject(response)}\n\n");
            }
            catch (Exception)
            {
                // ignored
            }
        }

        private static string RandomString(int length)
        {
            var random = new Random(Guid.NewGuid().GetHashCode());
            const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            return new string(Enumerable.Repeat(chars, length)
                .Select(s => s[random.Next(s.Length)]).ToArray());
        }

        private static string GetRequestBody(IDictionary<string, object> contextActionArguments)
        {
            return contextActionArguments.Aggregate(new StringBuilder(),
                (sb, kvp) => sb.AppendFormat("{0}{1} = {2}", sb.Length > 0 ? ", " : "", kvp.Key,
                    JsonConvert.SerializeObject(kvp.Value)),
                sb => sb.ToString());
        }
    }
}