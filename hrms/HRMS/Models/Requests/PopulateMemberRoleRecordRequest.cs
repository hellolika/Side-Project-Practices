using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class PopulateMemberRoleRecordRequest
{
    [JsonProperty("RoleId")]
    public int RoleId { get; set; }

    public void CheckModels()
    {
        if (string.IsNullOrEmpty(RoleId.ToString()))
        {
            throw new ApiException(ApiErrorEnum.InvalidModelState, "Role Id is required");
        }
    }
}