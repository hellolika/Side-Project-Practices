namespace HRMS.Models.Requests;

public class SendMessageBaseRequest
{
    public List<NotificationMessage> Messages { get; set; }

    public int MatchId { get; set; }

    public int MatchInfoId { get; set; }

    public int SubscriberType { get; set; }

    public DateTime? SendTime { get; set; }
}