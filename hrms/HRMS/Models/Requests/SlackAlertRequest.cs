using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class SlackAlertRequest
{
    [JsonProperty("text")]
    public string Text { get; set; }
}