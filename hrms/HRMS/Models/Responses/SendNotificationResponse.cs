namespace HRMS.Models.Responses;

public class SendNotificationResponse
{
    public string Id { get; set; }

    public int  Recipients { get; set; }

    public DateTime SentAt { get; set; }
}