using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class GetResignTransferRequest
{
    [JsonProperty("MemberId")] 
    public int MemberId { get; set; }
    
    [JsonProperty("StartDate")] 
    public DateTimeOffset StartDate { get; set; }

    [JsonProperty("ResignDate")] 
    public DateTimeOffset ResignDate { get; set; }
}