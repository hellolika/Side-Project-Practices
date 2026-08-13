using HRMS.Models.Requests;
using HRMS.Models.Responses;

namespace HRMS.Services.Interfaces;

public interface ISlackService
{
    Task<ApiBaseResponse<string>> SendSlackMessage(SlackAlertRequest slackAlertRequest);
    Task<ApiBaseResponse<SendSlackDirectMessageResponse>> SendSlackDirectMessage(SendSlackDirectMessageRequest slackAlertRequest);
    Task<ApiBaseResponse<SlackUserResponse>> GetAllSlackUsers();
    
    Task SendAbsenceAlert();
}