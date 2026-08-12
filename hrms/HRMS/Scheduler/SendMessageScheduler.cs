using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Services;
using HRMS.Services.Interfaces;

namespace HRMS.Scheduler;

public class SendMessageScheduler
{
    private readonly IBackOfficeService _backOfficeService;
    private readonly INotificationService _notificationService;
    private readonly ISlackService _slackService;

    public SendMessageScheduler(IBackOfficeService backOfficeService, INotificationService notificationService, ISlackService slackService)
    {
        _backOfficeService = backOfficeService;
        _notificationService = notificationService;
        _slackService = slackService;
    }
    public void RunJob()
    {
        var member = _backOfficeService.GetMembers();
        var filteredProbation = member.Result.Where(m => m.IsInProbation).ToList();

        foreach (var m in filteredProbation)
        {
            
            if (IsOneDayBeforeProbation(m.JoinDate) & !m.IsAlertProbation)
            {
                var editMemberInfo = new EditMemberInfoRequest() 
                {
                    IsResigned = m.IsResigned,
                    MemberId = m.MemberId,
                    Username = m.Username,
                    Email = m.Email,
                    PhoneNumber = m.PhoneNumber,
                    Address =m.Address,
                    Salary = m.Salary,
                    Permission = m.Permission,
                    BankAccount = m.BankAccount,
                    IsInProbation = m.IsInProbation,
                    Remark = m.Remark,
                    TeamId = m.TeamId,
                    JobGrade = m.JobGrade,
                    WorkLocationId = m.WorkLocationId,
                    IsSupervisor = m.IsSupervisor,
                    JoinDate = m.JoinDate,
                    IsAlertProbation = true,
                    BankName = m.BankName,
                    Position = m.Position,
                    EmployeeId = m.EmployeeId,
                    IsManager = m.IsManager
                };
                var response = _backOfficeService.EditMember(editMemberInfo);
                if (response.IsSuccess())
                {
                    SendEmployeeProbationNotification(m);
                    SendAdminNotificationToSlack(m);
                }
            }
        }
    }
    
    #region PrivateMethod
    private bool IsOneDayBeforeProbation(DateTimeOffset date)
    {
        return date.CompareTo(date.AddMonths(3).AddDays(-1)) == 0;
    }

    private void SendAdminNotificationToSlack(GetMembersResponse member)
    {
        var slackRequest = new SlackAlertRequest()
        {
            Text = $"{member.Username} is one day before passing the probation.\nEmail: {member.Email}\nJoin date: {member.JoinDate}"
        };
        
        _slackService.SendSlackMessage(slackRequest);
    }
    
    private void SendEmployeeProbationNotification(GetMembersResponse member)
    {
        var notificationRequest = new SendNotificationByExternalIdRequest
        {
            ExternalIds = new List<string> { member.Email }
        };

        var notificationMessage = new NotificationMessage()
        {
            Title = "Congratulation",
            Content = "Today is just one day before you pass your probation.",
        };
        notificationRequest.Messages = new List<NotificationMessage> { notificationMessage };
        _notificationService.SendNotificationByExternalIds(notificationRequest);
    }
    
    #endregion PrivateMethod
    
}