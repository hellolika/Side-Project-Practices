CREATE TABLE [dbo].[Team]
(
	[TeamId] INT NOT NULL PRIMARY KEY IDENTITY(1,1),
	[DepartmentId] INT DEFAULT 0,
	[TeamName] NVARCHAR(50) NOT NULL,
	[StartTime] TIME(0) NOT NULL,
	[EndTime] TIME(0) NOT NULL,
	[TotalHour] DECIMAL(16,9) NOT NULL,
	[IsEnable] BIT DEFAULT 1,
	[CreatedOn] DATETIME NULL,
	[CreatedBy] INT NULL,
	[ModifiedOn] DATETIME NULL,
	[ModifiedBy] INT NULL
)
