using System;
namespace HRMS.Models.Requests
{
    public class SendNotificationToAllSubscriberRequest : SendMessageBaseRequest
    {
        public bool TestMode { get; set; } = false;
    }
}

