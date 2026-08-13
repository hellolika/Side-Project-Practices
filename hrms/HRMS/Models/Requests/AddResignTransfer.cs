using HRMS.Models.Responses;
using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class AddResignTransfer : ResignTransferResponse
{
    
    [JsonProperty("StartDate")]
    public DateTimeOffset StartDate { get; set; }
    
    [JsonProperty("ResignDate")]
    public DateTimeOffset ResignDate { get; set; }

}