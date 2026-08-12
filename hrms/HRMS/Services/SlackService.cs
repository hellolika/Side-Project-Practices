using HRMS.Enum;
using HRMS.Helper;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Models.Settings;
using HRMS.Repositories.Interfaces;
using HRMS.Services.Interfaces;
using Microsoft.Extensions.Options;

namespace HRMS.Services;

public class SlackService : ISlackService
{
    private readonly IHttpCallingHelper _http;
    private readonly IMemberService _memberService;
    private readonly AppSettings _appSettings;
    private readonly ILoggerService _loggerService;
    private readonly IMemberRepository _memberRepository;
    private readonly IBackOfficeRepository _backOfficeRepository;


    public SlackService(IHttpCallingHelper http, IMemberService memberService, IOptions<AppSettings> appSettings,
        ILoggerService loggerService, IMemberRepository memberRepository, IBackOfficeRepository backOfficeRepository)
    {
        _http = http ;
        _memberService = memberService;
        _appSettings = appSettings.Value;
        _loggerService = loggerService;
        _memberRepository = memberRepository;
        _backOfficeRepository = backOfficeRepository;
    }
    
    public async Task<ApiBaseResponse<string>> SendSlackMessage(SlackAlertRequest slackAlertRequest)
    {
        await _http.PostCalling<string, SlackAlertRequest>(
             null, "SlackWebhookUrl",
            slackAlertRequest, EnumHttpContentType.Json);
        return new ApiBaseResponse<string>();
    }

    public async Task<ApiBaseResponse<SlackUserResponse>> GetAllSlackUsers()
    {
        var headers = new Dictionary<string, string>
        {
            { "Authorization", $"Bearer {_appSettings.SlackBotToken}" }
        };
        var response = await _http.GetCalling<SlackUserResponse>("https://slack.com/api/users.list", headers);
        var filteredMembers = response.Members.Where(member => member.Deleted == false  && member.IsBot == false).ToList();
        response.Members = filteredMembers;
        _memberService.UpsertSlackUsers(filteredMembers);
        return new ApiBaseResponse<SlackUserResponse>(response);
    }
    
    public async Task<ApiBaseResponse<SendSlackDirectMessageResponse>> SendSlackDirectMessage(SendSlackDirectMessageRequest slackAlertRequest)
    {
        var response = await _http.PostCalling<SendSlackDirectMessageResponse, SendSlackDirectMessageRequest>(_appSettings.SlackBotToken, "https://slack.com/api/chat.postMessage", slackAlertRequest, EnumHttpContentType.Json);
        return new ApiBaseResponse<SendSlackDirectMessageResponse>(response);
    }
    
    public async Task SendAbsenceAlert()
    {
        try
        {
            var date = DateTimeOffset.UtcNow.AddHours(7).Date;
            var absentMembers = _backOfficeRepository.GetAllAttendance(new GetAllAttendanceRequest()
            {
                Date = date
            }).Where(i => i.ClockIn == null && i.ClockOut == null).ToList();
            
            // Group absent members by their manager
            var managerAbsentMembers = absentMembers
                .SelectMany(absentMember => _memberRepository.GetDepartmentManager(absentMember.MemberId)
                    .Select(manager => new { Manager = manager, AbsentMember = absentMember }))
                .GroupBy(x => x.Manager.SlackId)
                .ToList();

            // Send notifications
            foreach (var group in managerAbsentMembers)
            {
                if (!string.IsNullOrEmpty(group.Key))
                {
                    var absentMemberNames = string.Join(", ", group.Select(x => string.IsNullOrEmpty(x.AbsentMember.SlackId) 
                        ? x.AbsentMember.Username 
                        : $"<@{x.AbsentMember.SlackId}|cal>"));
                    var text = $"Good Morning <@{group.Key}|cal>! The following {group.Count()} members are absent on {date.ToString("yyyy-MM-dd")}: {absentMemberNames}";
                    var managerSlackMessageRequest = new SendSlackDirectMessageRequest()
                    {
                        Channel = group.Key,
                        Text = text
                    };
                    await SendSlackDirectMessage(managerSlackMessageRequest);
                }
            }

            foreach (var absentMember in absentMembers)
            {
                if (!string.IsNullOrEmpty(absentMember.SlackId))
                {
                    var memberSlackMessageRequest = new SendSlackDirectMessageRequest()
                    {
                        Channel = absentMember.SlackId,
                        Text = $"Good Morning <@{absentMember.SlackId}|cal>! You are marked absent on {date.ToString("yyyy-MM-dd")}"
                    };
                    await SendSlackDirectMessage(memberSlackMessageRequest);
                }
            }

            _loggerService.Info("Scheduled to send notification to all absent members and their managers");
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            throw;
        }
    } 
}