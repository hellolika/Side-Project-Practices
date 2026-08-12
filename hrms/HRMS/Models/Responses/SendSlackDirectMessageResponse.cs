using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class SendSlackDirectMessageResponse
{
    [JsonProperty("ok")]
    public bool Ok { get; set; }
}