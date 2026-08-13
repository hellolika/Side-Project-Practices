using HRMS.Models.Requests;
using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class SendSlackDirectMessageRequest: SlackAlertRequest
{
    [JsonProperty("channel")]
    public string Channel { get; set; }
}