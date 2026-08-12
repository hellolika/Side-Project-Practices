using HRMS.Models.Requests;

namespace HRMS.Models;

public class SendNotificationByExternalIdRequest : SendMessageBaseRequest
{
    public List<string> ExternalIds { get; set; }
}