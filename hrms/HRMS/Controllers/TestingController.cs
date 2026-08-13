using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Models.Settings;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using Newtonsoft.Json;

namespace HRMS.Controllers
{
    [Route("/api/[controller]")]
    [ApiController]
    public class TestingController : Controller
    {
        private readonly INotificationService _notificationService;
        private readonly ILoggerService _loggerService;
        private readonly ISlackService _slackService;
        private readonly AppSettings _appSettings;


        public TestingController(INotificationService notificationService, ILoggerService loggerService, 
            ISlackService slackService, IOptions<AppSettings> appSettings)
        {
            _notificationService = notificationService;
            _loggerService = loggerService;
            _slackService = slackService;
            _appSettings = appSettings.Value;
        }


        [HttpGet("TestCalling")]
        public ApiBaseResponse<string> TestCalling()
        {
            return new ApiBaseResponse<string>("Get Test String!");
        }
        
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpGet("GetAllSlackUsers")]
        public async Task<ApiBaseResponse<SlackUserResponse>> GetAllSlackUsers()
        {
            return await _slackService.GetAllSlackUsers();
        }
        
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpPost("SendSlackMessage")]
        public async Task<ApiBaseResponse<SendSlackDirectMessageResponse>> SendSlackMessage(SendSlackDirectMessageRequest slackAlertRequest)
        {
            return await _slackService.SendSlackDirectMessage(slackAlertRequest);
        }
        
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpPost("GetSlackToken")]
        public async Task<ApiBaseResponse<string>> SendSlackMessageV2()
        {
            return new ApiBaseResponse<string>(_appSettings.SlackBotToken);
        }
        
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpGet("ManualSendAbsenceAlert")]
        public async Task<ApiBaseResponse<string>> ManualSendAbsenceAlert()
        {
            _loggerService.Info($"manual to send absence alert to all member on {DateTime.Now}");
            await _slackService.SendAbsenceAlert();
            return new ApiBaseResponse<string>("sent");
        }
    }
}